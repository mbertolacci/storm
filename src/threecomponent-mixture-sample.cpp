#include <Rcpp.h>
#include "mixture-utils.h"
#include "gev.h"
#include "threecomponent-mixture-utils.h"

using namespace Rcpp;

void sampleP(NumericVector pCurrent, NumericVector nCurrent, NumericVector pPrior) {
    int k = pCurrent.length();
    double sum = 0;
    for (int j = 0; j < k; ++j) {
        pCurrent[j] = R::rgamma(pPrior[j] + nCurrent[j], 1);
        sum += pCurrent[j];
    }
    for (int j = 0; j < k; ++j) {
        pCurrent[j] /= sum;
    }
}

void sampleZ(NumericVector zCurrent, NumericVector pCurrent, ThetaValues thetaCurrent, NumericVector y) {
    int n = y.length();

    double pGammaOrGEV = pCurrent[1] + pCurrent[2];
    double pGamma = pCurrent[1] / pGammaOrGEV;
    double pGev = pCurrent[2] / pGammaOrGEV;

    double gammaNorm = pow(thetaCurrent.beta, -thetaCurrent.alpha) / tgamma(thetaCurrent.alpha);

    #pragma omp parallel for simd
    for (int i = 0; i < n; ++i) {
        if (y[i] == 0) {
            zCurrent[i] = 1;
        } else if (!isInGEVSupport(y[i], thetaCurrent)) {
            zCurrent[i] = 2;
        } else {
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

// [[Rcpp::export(name=".threeComponentMixtureSample")]]
List threeComponentMixtureSample(
    int nSamples, int burnIn, NumericVector y, List prior, NumericVector zStart, NumericVector thetaStart
) {
    int nIterations = nSamples + burnIn;
    int k = 3;

    NumericVector pPrior = prior["theta.p"];
    double alphaPrior = prior["theta.alpha"];
    double betaPrior = prior["theta.beta"];
    double xiPrior = prior["theta.xi"];
    double sigmaPrior = prior["theta.sigma"];
    double muPrior = prior["theta.mu"];
    // double sigmaMuRho = prior["theta.mu.sigma.rho"];

    int n = y.length();
    NumericVector logY = log(y);

    NumericVector zCurrent = clone(zStart);

    ThetaValues thetaCurrent(thetaStart);

    NumericVector pCurrent(k);
    NumericVector nCurrent(k);

    NumericMatrix thetaSample(nSamples, k + 5);
    NumericMatrix zSample(nSamples, n);
    NumericVector ySample(nSamples);
    NumericVector ySampleZ(nSamples);

    for (int iteration = 0; iteration < nIterations; ++iteration) {
        nCurrent.fill(0);

        double sumGammaY = 0;
        double sumLogGammaY = 0;
        for (int i = 0; i < n; ++i) {
            int z = zCurrent[i] - 1;
            nCurrent[z]++;

            if (z == 1) {
                sumGammaY += y[i];
                sumLogGammaY += logY[i];
            }
        }

        thetaCurrent.gammaLogDensity = gammaLogDensity(thetaCurrent, sumGammaY, sumLogGammaY, nCurrent[1]);
        thetaCurrent.gevLogDensity = gevLogDensity(thetaCurrent, y, zCurrent);

        sampleP(pCurrent, nCurrent, pPrior);
        thetaCurrent = sampleAlpha(thetaCurrent, sumGammaY, sumLogGammaY, nCurrent[1], alphaPrior);
        thetaCurrent = sampleBeta(thetaCurrent, sumGammaY, sumLogGammaY, nCurrent[1], betaPrior);
        thetaCurrent = sampleXi(thetaCurrent, y, zCurrent, xiPrior);
        // thetaCurrent = sampleMuAndSigma(thetaCurrent, y, zCurrent, muPrior, sigmaPrior, sigmaMuRho);
        thetaCurrent = sampleSigma(thetaCurrent, y, zCurrent, sigmaPrior);
        thetaCurrent = sampleMu(thetaCurrent, y, zCurrent, muPrior);

        sampleZ(zCurrent, pCurrent, thetaCurrent, y);

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
            thetaSample(index, 5) = pCurrent[0];
            thetaSample(index, 6) = pCurrent[1];
            thetaSample(index, 7) = pCurrent[2];

            zSample.row(index) = zCurrent;

            int z = 1 + randomWeightedIndex(pCurrent);
            ySample[index] = sampleYGivenZ(thetaCurrent, z);
            ySampleZ[index] = z;
        }
    }

    thetaCurrent.printAcceptanceRatios(nSamples);

    List results;
    results["theta.sample"] = thetaSample;
    results["z.sample"] = zSample;
    results["y.sample"] = ySample;
    results["y.sample.z"] = ySampleZ;
    results["n.params"] = 5;
    results["k"] = k;
    results["identification.method"] = "none";

    return results;
}
