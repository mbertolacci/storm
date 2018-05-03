#include <algorithm>
#include <chrono>
#include <vector>

#include <RcppArmadillo.h>

#include "distribution.hpp"
#include "parameter-sampler.hpp"
#include "rng.hpp"
#include "utils.hpp"

using arma::colvec;
using arma::mat;
using arma::ucolvec;
using arma::umat;

using Rcpp::as;
using Rcpp::checkUserInterrupt;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericVector;
using Rcpp::Rcout;
using Rcpp::StringVector;
using Rcpp::wrap;

using ptsm::RNG;
using ptsm::rng;

mat samplePHMM(ucolvec zCurrent, ucolvec zPrevious, mat pPrior) {
    mat P(size(pPrior));
    colvec n(pPrior.n_cols);

    for (unsigned int j = 0; j < P.n_rows; ++j) {
        n.fill(0);
        for (unsigned int i = 0; i < zCurrent.n_elem; ++i) {
            if (zPrevious[i] == j + 1) {
                ++n[zCurrent[i] - 1];
            }
        }

        double sum = 0;
        for (unsigned int jj = 0; jj < P.n_cols; ++jj) {
            P(j, jj) = rng.randg(pPrior(j, jj) + n[jj], 1);
            sum += P(j, jj);
        }
        for (unsigned int jj = 0; jj < P.n_cols; ++jj) {
            P(j, jj) /= sum;
        }
    }

    return P;
}

unsigned int sampleSingleZ2(const colvec &p) {
    double u = rng.randu();
    colvec pNormalised = p / arma::sum(p);
    for (unsigned int k = 0; k < pNormalised.n_elem; ++k) {
        if (u < pNormalised[k]) {
            return k + 1;
        }
        u -= pNormalised[k];
    }
    return pNormalised.n_elem - 1;
}

// [[Rcpp::export(name=".hmm_sample")]]
List hmmSample(
    unsigned int nSamples, unsigned int burnIn,
    NumericVector yR,
    StringVector distributionNames, List priors, List samplingSchemes,
    IntegerVector zStart, List thetaStart,
    int thetaSampleThinning = 1, int zSampleThinning = 0, int yMissingSampleThinning = 0,
    unsigned int verbose = 0
) {
    RNG::initialise();

    unsigned int nIterations = nSamples + burnIn;

    // Priors and samplers
    Distribution lowerDistribution(distributionNames[0]);
    Distribution upperDistribution(distributionNames[1]);

    ParameterSampler lowerSampler(priors[0], samplingSchemes[0], lowerDistribution);
    ParameterSampler upperSampler(priors[1], samplingSchemes[1], upperDistribution);

    mat pPrior = as<mat>(priors["P"]);

    // Data and missing data
    colvec y = as<colvec>(yR);
    colvec logY = log(y);

    MissingValuesPair missingValuesPair = findMissingValues(yR);
    ucolvec yMissingIndices = missingValuesPair.first;
    std::vector<bool> yIsMissing = missingValuesPair.second;

    // Starting values for parameters
    ucolvec zCurrent = as<ucolvec>(zStart);
    ucolvec zPrevious(zCurrent.n_elem);
    std::copy(zCurrent.begin(), zCurrent.end() - 1, zPrevious.begin() + 1);

    colvec thetaLower = as<colvec>(thetaStart[0]);
    colvec thetaUpper = as<colvec>(thetaStart[1]);
    mat pCurrent(3, 3);
    mat pCurrentGivenZ(y.n_elem, 2);

    // Samples for output
    mat thetaSample;
    if (thetaSampleThinning > 0) {
        thetaSample = mat(
            ceil(static_cast<double>(nSamples) / static_cast<double>(thetaSampleThinning)),
            thetaLower.n_elem + thetaUpper.n_elem + 9
        );
    }
    ucolvec z0Sample;
    umat zSample;
    if (zSampleThinning > 0) {
        int nZSamples = ceil(static_cast<double>(nSamples) / static_cast<double>(zSampleThinning));
        z0Sample = ucolvec(nZSamples);
        zSample = umat(nZSamples, y.n_elem);
    }
    mat yMissingSample;
    if (yMissingSampleThinning > 0) {
        int nYMissingSamples = ceil(static_cast<double>(nSamples) / static_cast<double>(yMissingSampleThinning));
        yMissingSample = mat(nYMissingSamples, yMissingIndices.size());
    }

    // Initial missing value sample
    sampleMissingY(
        y, logY, yMissingIndices, zCurrent,
        {
            ParameterBoundDistribution(thetaLower, lowerDistribution),
            ParameterBoundDistribution(thetaUpper, upperDistribution)
        }
    );

    // Initial z0 sample
    zPrevious[0] = sampleSingleZ({ pCurrent(zCurrent[0] - 1, 1), pCurrent(zCurrent[0] - 1, 2) });

    std::chrono::time_point<std::chrono::system_clock> startIteration, endIteration;
    startIteration = std::chrono::system_clock::now();

    for (unsigned int iteration = 0; iteration < nIterations; ++iteration) {
        // Sample p
        pCurrent = samplePHMM(zCurrent, zPrevious, pPrior);

        // Sample mixture 2
        thetaLower = lowerSampler.sample(thetaLower, DataBoundDistribution(y, logY, zCurrent, 2, lowerDistribution));

        // Sample mixture 3
        thetaUpper = upperSampler.sample(thetaUpper, DataBoundDistribution(y, logY, zCurrent, 3, upperDistribution));

        // Sample \bm{z}
        std::vector<ParameterBoundDistribution> boundDistributions = {
            ParameterBoundDistribution(thetaLower, lowerDistribution),
            ParameterBoundDistribution(thetaUpper, upperDistribution)
        };
        // Compute p(z|y_{0,...,t}, \theta), not including missing
        mat pMarginal(y.n_elem, 3, arma::fill::zeros);
        for (unsigned int i = 0; i < zCurrent.n_elem; ++i) {
            if (y[i] == 0 && !yIsMissing[i]) {
                pMarginal(i, 0) = 1;
            } else {
                for (unsigned int toState = 0; toState < 3; ++toState) {
                    for (unsigned int fromState = 0; fromState < 3; ++fromState) {
                        if (i == 0) {
                            pMarginal(i, toState) += pCurrent(fromState, toState) / 3.0;
                        } else {
                            pMarginal(i, toState) += pCurrent(fromState, toState) * pMarginal(i - 1, fromState);
                        }
                    }
                }
                if (!yIsMissing[i]) {
                    // We know at this point that y is not 0
                    pMarginal(i, 0) = 0;
                    for (unsigned int state = 1; state < 3; ++state) {
                        pMarginal(i, state) *= boundDistributions[state - 1].pdf(y[i]);
                    }
                    pMarginal.row(i) /= arma::sum(pMarginal.row(i));
                }
            }
        }
        zCurrent[zCurrent.n_elem - 1] = sampleSingleZ2(pMarginal.row(zCurrent.n_elem - 1).t());
        for (int i = zCurrent.n_elem - 2; i >= 0; --i) {
            zCurrent[i] = sampleSingleZ2(
                pMarginal.row(i).t() % pCurrent.col(zCurrent[i + 1] - 1)
            );
        }
        zPrevious[0] = sampleSingleZ2(pCurrent.col(zCurrent[0] - 1) / 3.0);

        std::copy(zCurrent.begin(), zCurrent.end() - 1, zPrevious.begin() + 1);

        // Sample missing values \bm{y^*}
        sampleMissingY(
            y, logY, yMissingIndices, zCurrent,
            {
                ParameterBoundDistribution(thetaLower, lowerDistribution),
                ParameterBoundDistribution(thetaUpper, upperDistribution)
            }
        );

        if (verbose > 0 && iteration % verbose == 0) {
            Rcout << "in " << iteration << " have "
                << thetaLower.t() << " | "
                << thetaUpper.t() << " | "
                << pCurrent << "\n";

            endIteration = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = endIteration - startIteration;
            double timePerIteration = (1000 * elapsed_seconds.count() / verbose);
            Rcout << "time per iteration is " << timePerIteration << "ms ";
            Rcout << "(estimated time remaining " << ((nIterations - iteration) * timePerIteration) / 1000 << "s)\n";
            startIteration = endIteration;

            // NOTE(mgnb): checks whether the user has pressed Ctrl-C (among other things)
            checkUserInterrupt();
        }

        if (verbose > 0 && iteration == burnIn) {
            Rcout << "Burn in complete\n";
        }

        if (iteration >= burnIn) {
            int index = iteration - burnIn;

            if (thetaSampleThinning > 0 && (index % thetaSampleThinning == 0)) {
                copyMultiple(
                    thetaSample.begin_row(index / thetaSampleThinning),
                    thetaLower, thetaUpper,
                    pCurrent.t().eval()  // Load P row-wise
                );
            }

            if (zSampleThinning > 0 && (index % zSampleThinning == 0)) {
                unsigned int zSampleIndex = index / zSampleThinning;
                z0Sample[zSampleIndex] = zPrevious[0];
                zSample.row(zSampleIndex) = zCurrent.t();
            }

            if (yMissingSampleThinning > 0 && (index % yMissingSampleThinning == 0)) {
                yMissingSample.row(index / yMissingSampleThinning) = y(yMissingIndices).t();
            }
        }
    }

    List results;
    if (thetaSampleThinning > 0) {
        results["theta_sample"] = wrap(thetaSample);
    }
    if (zSampleThinning > 0) {
        results["z0_sample"] = wrap(z0Sample);
        results["z_sample"] = wrap(zSample);
    }
    if (yMissingSampleThinning > 0) {
        results["y_missing_sample"] = wrap(yMissingSample);
    }

    return results;
}
