#include <chrono>
#include <RcppArmadillo.h>
#include <vector>

#include "distribution.hpp"
#include "parameter-sampler.hpp"
#include "rgamma-thread-safe.hpp"
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

colvec sampleP(ucolvec zCurrent, colvec pPrior) {
    colvec p(pPrior.n_elem);
    colvec n(pPrior.n_elem, arma::fill::zeros);

    for (unsigned int i = 0; i < zCurrent.n_elem; ++i) {
        ++n[zCurrent[i] - 1];
    }

    double sum = 0;
    for (unsigned int j = 0; j < p.n_elem; ++j) {
        p[j] = rgammaThreadSafe(pPrior[j] + n[j], 1);
        sum += p[j];
    }
    for (unsigned int j = 0; j < p.n_elem; ++j) {
        p[j] /= sum;
    }

    return p;
}

// [[Rcpp::export(name=".ptsm_independent_sample")]]
List independentSample(
    unsigned int nSamples, unsigned int burnIn,
    NumericVector yR,
    StringVector distributionNames, List priors, List samplingSchemes,
    IntegerVector zStart, List thetaStart,
    int thetaSampleThinning = 1, int zSampleThinning = 0, int yMissingSampleThinning = 0,
    unsigned int verbose = 0
) {
    unsigned int nIterations = nSamples + burnIn;

    // Priors and samplers
    Distribution lowerDistribution(distributionNames[0]);
    Distribution upperDistribution(distributionNames[1]);
    ParameterSampler lowerSampler(priors[0], samplingSchemes[0], lowerDistribution);
    ParameterSampler upperSampler(priors[1], samplingSchemes[1], upperDistribution);

    colvec pPrior = as<colvec>(priors["p"]);

    // Data and missing data
    colvec y = as<colvec>(yR);
    colvec logY = log(y);

    MissingValuesPair missingValuesPair = findMissingValues(yR);
    ucolvec yMissingIndices = missingValuesPair.first;
    std::vector<bool> yIsMissing = missingValuesPair.second;

    // Starting values for parameters
    ucolvec zCurrent = as<ucolvec>(zStart);
    colvec thetaLower = as<colvec>(thetaStart[0]);
    colvec thetaUpper = as<colvec>(thetaStart[1]);
    colvec pCurrent(3);
    mat pCurrentGivenZ(y.n_elem, 2);

    // Samples for output
    mat thetaSample;
    if (thetaSampleThinning > 0) {
        thetaSample = mat(
            ceil(static_cast<double>(nSamples) / static_cast<double>(thetaSampleThinning)),
            thetaLower.n_elem + thetaUpper.n_elem + 3
        );
    }
    umat zSample;
    if (zSampleThinning > 0) {
        int nZSamples = ceil(static_cast<double>(nSamples) / static_cast<double>(zSampleThinning));
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
        ParameterBoundDistribution(thetaLower, lowerDistribution),
        ParameterBoundDistribution(thetaUpper, upperDistribution)
    );

    std::chrono::time_point<std::chrono::system_clock> startIteration, endIteration;
    startIteration = std::chrono::system_clock::now();

    for (unsigned int iteration = 0; iteration < nIterations; ++iteration) {
        // Sample p
        pCurrent = sampleP(zCurrent, pPrior);

        // Sample mixture 2
        thetaLower = lowerSampler.sample(thetaLower, DataBoundDistribution(y, logY, zCurrent, 2, lowerDistribution));

        // Sample mixture 3
        thetaUpper = upperSampler.sample(thetaUpper, DataBoundDistribution(y, logY, zCurrent, 3, upperDistribution));

        // Sample \bm{z}
        pCurrentGivenZ.col(0).fill(pCurrent[1]);
        pCurrentGivenZ.col(1).fill(pCurrent[2]);
        zCurrent = sampleZ(
            pCurrentGivenZ, y, yIsMissing,
            ParameterBoundDistribution(thetaLower, lowerDistribution),
            ParameterBoundDistribution(thetaUpper, upperDistribution)
        );

        // Sample missing values \bm{y^*}
        sampleMissingY(
            y, logY, yMissingIndices, zCurrent,
            ParameterBoundDistribution(thetaLower, lowerDistribution),
            ParameterBoundDistribution(thetaUpper, upperDistribution)
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
            lowerSampler.printAcceptanceRatio(burnIn);
            upperSampler.printAcceptanceRatio(burnIn);
            lowerSampler.resetAcceptCount();
            upperSampler.resetAcceptCount();
        }

        if (iteration >= burnIn) {
            int index = iteration - burnIn;

            if (thetaSampleThinning > 0 && (index % thetaSampleThinning == 0)) {
                copyMultiple(
                    thetaSample.begin_row(index / thetaSampleThinning),
                    thetaLower, thetaUpper, pCurrent
                );
            }

            if (zSampleThinning > 0 && (index % zSampleThinning == 0)) {
                zSample.row(index / zSampleThinning) = zCurrent.t();
            }

            if (yMissingSampleThinning > 0 && (index % yMissingSampleThinning == 0)) {
                yMissingSample.row(index / yMissingSampleThinning) = y(yMissingIndices).t();
            }
        }
    }

    if (verbose > 0) {
        lowerSampler.printAcceptanceRatio(nSamples);
        upperSampler.printAcceptanceRatio(nSamples);
    }

    List results;
    if (thetaSampleThinning > 0) {
        results["theta_sample"] = wrap(thetaSample);
    }
    if (zSampleThinning > 0) {
        results["z_sample"] = wrap(zSample);
    }
    if (yMissingSampleThinning > 0) {
        results["y_missing_sample"] = wrap(yMissingSample);
    }

    return results;
}
