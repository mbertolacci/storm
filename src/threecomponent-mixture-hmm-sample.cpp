#include <Rcpp.h>
#include "mixture-utils.h"
#include "gev.h"
#include "threecomponent-mixture-utils.h"

using namespace Rcpp;

void sampleMissingY(NumericVector y, NumericVector logY, std::vector<int> yMissingIndices, NumericVector zCurrent, ThetaValues thetaCurrent) {
    int nMissing = yMissingIndices.size();
    #pragma omp parallel for simd
    for (int i = 0; i < nMissing; ++i) {
        int index = yMissingIndices[i];
        y[index] = sampleYGivenZ(thetaCurrent, zCurrent[index]);
        logY[index] = log(y[index]);
    }
}

void sampleP(NumericMatrix pCurrent, NumericMatrix nCurrent, NumericMatrix pPrior) {
    int k = 3;
    for (int j = 0; j < k; ++j) {
        double sum = 0;
        for (int jj = 0; jj < k; ++jj) {
            pCurrent(j, jj) = R::rgamma(pPrior(j, jj) + nCurrent(j, jj), 1);
            sum += pCurrent(j, jj);
        }

        for (int jj = 0; jj < k; ++jj) {
            pCurrent(j, jj) /= sum;
        }
    }
}

void sampleZ(
    NumericVector zCurrent, NumericVector zPrev,
    NumericMatrix pCurrent, ThetaValues thetaCurrent,
    NumericVector y, const std::vector<bool> yIsMissing
) {
    int n = y.length();

    double gammaNorm = pow(thetaCurrent.beta, -thetaCurrent.alpha) / tgamma(thetaCurrent.alpha);

    // NOTE(mgnb): caching this here so we don't use zCurrent[1] in the loop below *after* it's been updated
    int z2 = zCurrent[1];

    // NOTE(mgnb): we have to use zPrev here instead of zCurrent because the parallel loop might use a new
    // zCurrent[i - 1] value. This is still drawing from the conditional distribution, it's equivalent to drawing
    // in the order z_n, z_{n-1}, ..., z_1, rather than the more natural z_1, z_2, ..., z_n
    #pragma omp parallel for simd
    for (int i = 0; i < n; ++i) {
        if (yIsMissing[i]) {
            // Missing value: sample z based on previous z
            int zMinusOne = zPrev[i] - 1;
            double u = R::unif_rand();
            for (int j = 0; j < 3; ++j) {
                zCurrent[i] = j + 1;
                u -= pCurrent(zMinusOne, j);
                if (u < 0) {
                    break;
                }
            }
        } else {
            if (y[i] == 0) {
                zCurrent[i] = 1;
            } else if (!isInGEVSupport(y[i], thetaCurrent)) {
                zCurrent[i] = 2;
            } else {
                double pGammaOrGEV;
                double pGamma;
                double pGev;
                if (i == 0) {
                    // p(z_1|z_2) \propto p(z_2|z_1)p(z_1)
                    // Then, if p(z_1) = 1/3, we have
                    // p(z_1|z_2) \propto p(z_2|z_1) = P_{z_1z_2}
                    pGammaOrGEV = pCurrent(1, z2) + pCurrent(2, z2);
                    pGamma = pCurrent(1, z2) / pGammaOrGEV;
                    pGev = pCurrent(2, z2) / pGammaOrGEV;
                } else {
                    int zMinusOne = zPrev[i] - 1;
                    pGammaOrGEV = pCurrent(zMinusOne, 1) + pCurrent(zMinusOne, 2);
                    pGamma = pCurrent(zMinusOne, 1) / pGammaOrGEV;
                    pGev = pCurrent(zMinusOne, 2) / pGammaOrGEV;
                }

                // NOTE(mike): use this instead of R::dgamma because precomputing the normalising constant saves a lot of time
                double gammaDensity = pGamma * gammaNorm * pow(y[i], thetaCurrent.alpha - 1) * exp(-y[i] / thetaCurrent.beta);
                double gevDensity = pGev * dgev(
                    y[i], thetaCurrent.mu, thetaCurrent.sigma, thetaCurrent.xi, false
                );

                if (R::runif(0, gammaDensity + gevDensity) < gammaDensity) {
                    zCurrent[i] = 2;
                } else {
                    zCurrent[i] = 3;
                }
            }
        }
    }

    for (int i = 1; i < n; ++i) {
        zPrev[i] = zCurrent[i - 1];
    }
}

// [[Rcpp::export(name=".threeComponentMixtureHMMSample")]]
List threeComponentMixtureHMMSample(
    int nSamples, int burnIn, NumericVector y, List prior, NumericVector zStart, NumericVector thetaStart
) {
    int nIterations = nSamples + burnIn;
    int k = 3;

    y = clone(y);

    NumericMatrix pPrior = prior["theta.p"];
    double alphaPrior = prior["theta.alpha"];
    double betaPrior = prior["theta.beta"];
    double xiPrior = prior["theta.xi"];
    double sigmaPrior = prior["theta.sigma"];
    double muPrior = prior["theta.mu"];
    // NumericMatrix jointCholeskyPrior = prior["theta.joint.cholesky"];

    int n = y.length();
    std::vector<int> yMissingIndices(n);
    std::vector<bool> yIsMissing(n);
    int nMissing = 0;
    for (int i = 0; i < n; ++i) {
        if (NumericVector::is_na(y[i])) {
            yMissingIndices[nMissing] = i;
            yIsMissing[i] = true;
            ++nMissing;
        } else {
            yIsMissing[i] = false;
        }
    }
    yMissingIndices.resize(nMissing);
    NumericVector logY = log(y);

    NumericVector zCurrent = clone(zStart);
    NumericVector zPrev(n);
    for (int i = 1; i < n; ++i) {
        zPrev[i] = zCurrent[i - 1];
    }

    ThetaValues thetaCurrent(thetaStart);

    NumericMatrix pCurrent(k, k);
    NumericMatrix nCurrent(k, k);

    NumericMatrix thetaSample(nSamples, 5 + k * k);
    NumericMatrix zSample(nSamples, n);
    NumericVector ySample(nSamples);
    NumericVector ySampleZ(nSamples);

    NumericMatrix yMissingSample(nSamples, nMissing);

    for (int iteration = 0; iteration < nIterations; ++iteration) {
        nCurrent.fill(0);
        for (int i = 1; i < n; ++i) {
            nCurrent(zPrev[i] - 1, zCurrent[i] - 1)++;
        }

        sampleMissingY(y, logY, yMissingIndices, zCurrent, thetaCurrent);

        double sumGammaY = 0;
        double sumLogGammaY = 0;
        int nGamma = 0;
        #pragma omp parallel for simd reduction(+:sumGammaY) reduction(+:sumLogGammaY) reduction(+:nGamma)
        for (int i = 0; i < n; ++i) {
            if (zCurrent[i] == 2) {
                sumGammaY += y[i];
                sumLogGammaY += logY[i];
                ++nGamma;
            }
        }

        thetaCurrent.gammaLogDensity = gammaLogDensity(thetaCurrent, sumGammaY, sumLogGammaY, nGamma);
        thetaCurrent.gevLogDensity = gevLogDensity(thetaCurrent, y, zCurrent);

        sampleP(pCurrent, nCurrent, pPrior);
        // thetaCurrent = sampleJoint(thetaCurrent, y, sumGammaY, sumLogGammaY, nGamma, zCurrent, jointCholeskyPrior);
        thetaCurrent = sampleAlpha(thetaCurrent, sumGammaY, sumLogGammaY, nGamma, alphaPrior);
        thetaCurrent = sampleBeta(thetaCurrent, sumGammaY, sumLogGammaY, nGamma, betaPrior);
        thetaCurrent = sampleXi(thetaCurrent, y, zCurrent, xiPrior);
        thetaCurrent = sampleSigma(thetaCurrent, y, zCurrent, sigmaPrior);
        thetaCurrent = sampleMu(thetaCurrent, y, zCurrent, muPrior);

        sampleZ(zCurrent, zPrev, pCurrent, thetaCurrent, y, yIsMissing);

        if (iteration % 500 == 0) {
            Rcout << "in " << iteration << " have "
                << nCurrent
                << " | "
                << thetaCurrent.alpha << " "
                << thetaCurrent.beta << " | "
                << thetaCurrent.mu << " "
                << thetaCurrent.sigma << " "
                << thetaCurrent.xi << " "
                << "\n";

            thetaCurrent.printGEVSupport();
            // NOTE(mgnb): checks whether the user has pressed Ctrl-C (among other things)
            checkUserInterrupt();
        }

        if (iteration == burnIn) {
            Rcout << "Burn in complete\n";
            thetaCurrent.printAcceptanceRatios(burnIn);
            thetaCurrent.resetAcceptCounts();
        }

        if (iteration >= burnIn) {
            int index = iteration - burnIn;
            thetaSample(index, 0) = thetaCurrent.alpha;
            thetaSample(index, 1) = thetaCurrent.beta;
            thetaSample(index, 2) = thetaCurrent.mu;
            thetaSample(index, 3) = thetaCurrent.sigma;
            thetaSample(index, 4) = thetaCurrent.xi;
            thetaSample(index, 5) = pCurrent(0, 0);
            thetaSample(index, 6) = pCurrent(0, 1);
            thetaSample(index, 7) = pCurrent(0, 2);
            thetaSample(index, 8) = pCurrent(1, 0);
            thetaSample(index, 9) = pCurrent(1, 1);
            thetaSample(index, 10) = pCurrent(1, 2);
            thetaSample(index, 11) = pCurrent(2, 0);
            thetaSample(index, 12) = pCurrent(2, 1);
            thetaSample(index, 13) = pCurrent(2, 2);

            zSample.row(index) = zCurrent;
            for (int i = 0; i < nMissing; ++i) {
                yMissingSample(index, i) = y[yMissingIndices[i]];
            }

            int prevZ = 0;
            if (index > 0) {
                prevZ = ySampleZ[index - 1] - 1;
            }
            int z = 1 + randomWeightedIndex(pCurrent.row(prevZ));
            ySample[index] = sampleYGivenZ(thetaCurrent, z);
            ySampleZ[index] = z;
        }
    }

    thetaCurrent.printAcceptanceRatios(nSamples);

    List results;
    results["theta.sample"] = thetaSample;
    results["y.missing.sample"] = yMissingSample;
    results["z.sample"] = zSample;
    results["y.sample"] = ySample;
    results["y.sample.z"] = ySampleZ;
    results["n.params"] = 5;
    results["k"] = k;
    results["identification.method"] = "none";

    return results;
}
