#include <RcppArmadillo.h>
#include <vector>

#include "distribution.hpp"
#include "parameter-sampler.hpp"
#include "rng.hpp"
#include "utils.hpp"
#include "progress.hpp"

#include "independent-sampler.hpp"

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

colvec sampleP(ucolvec zCurrent, colvec pPrior) {
    colvec p(pPrior.n_elem);
    colvec n(pPrior.n_elem, arma::fill::zeros);

    for (unsigned int i = 0; i < zCurrent.n_elem; ++i) {
        ++n[zCurrent[i] - 1];
    }

    double sum = 0;
    for (unsigned int j = 0; j < p.n_elem; ++j) {
        p[j] = rng.randg(pPrior[j] + n[j], 1);
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
    IntegerVector zStart, List distributionsStart,
    unsigned int distributionSampleThinning = 1, unsigned int pSampleThinning = 1,
    unsigned int zSampleThinning = 0, unsigned int yMissingSampleThinning = 0,
    bool progress = false
) {
    RNG::initialise();

    unsigned int nIterations = nSamples + burnIn;

    // Priors and samplers
    Distribution lowerDistribution(distributionNames[0]);
    Distribution upperDistribution(distributionNames[1]);
    List distributionPriors = priors["distributions"];
    List distributionSamplingSchemes = samplingSchemes["distributions"];
    ParameterSampler lowerSampler(distributionPriors[0], distributionSamplingSchemes[0], lowerDistribution);
    ParameterSampler upperSampler(distributionPriors[1], distributionSamplingSchemes[1], upperDistribution);

    colvec pPrior = as<colvec>(priors["p"]);

    // Data and missing data
    colvec y = as<colvec>(yR);
    colvec logY = log(y);

    MissingValuesPair missingValuesPair = findMissingValues(yR);
    ucolvec yMissingIndices = missingValuesPair.first;
    std::vector<bool> yIsMissing = missingValuesPair.second;

    // Starting values for parameters
    ucolvec zCurrent = as<ucolvec>(zStart);
    colvec thetaLower = as<colvec>(distributionsStart[0]);
    colvec thetaUpper = as<colvec>(distributionsStart[1]);
    colvec pCurrent(3);
    mat pCurrentGivenZ(y.n_elem, 2);

    // Samples for output
    mat lowerSample;
    mat upperSample;
    if (distributionSampleThinning > 0) {
        unsigned int nDistributionSamples = ceil(
            static_cast<double>(nSamples) / static_cast<double>(distributionSampleThinning)
        );
        lowerSample = mat(nDistributionSamples, thetaLower.n_elem);
        upperSample = mat(nDistributionSamples, thetaUpper.n_elem);
    }
    mat pSample;
    if (pSampleThinning > 0) {
        unsigned int nPSamples = ceil(
            static_cast<double>(nSamples) / static_cast<double>(pSampleThinning)
        );
        pSample = mat(nPSamples, 3);
    }
    umat zSample;
    if (zSampleThinning > 0) {
        unsigned int nZSamples = ceil(static_cast<double>(nSamples) / static_cast<double>(zSampleThinning));
        zSample = umat(nZSamples, y.n_elem);
    }
    mat yMissingSample;
    if (yMissingSampleThinning > 0) {
        unsigned int nYMissingSamples = ceil(static_cast<double>(nSamples) / static_cast<double>(yMissingSampleThinning));
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

    ProgressBar progressBar(nIterations);
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
            {
                ParameterBoundDistribution(thetaLower, lowerDistribution),
                ParameterBoundDistribution(thetaUpper, upperDistribution)
            }
        );

        // Sample missing values \bm{y^*}
        sampleMissingY(
            y, logY, yMissingIndices, zCurrent,
            {
                ParameterBoundDistribution(thetaLower, lowerDistribution),
                ParameterBoundDistribution(thetaUpper, upperDistribution)
            }
        );

        if (iteration >= burnIn) {
            int index = iteration - burnIn;

            if (distributionSampleThinning > 0 && (index % distributionSampleThinning == 0)) {
                unsigned int sampleIndex = index / distributionSampleThinning;
                lowerSample.row(sampleIndex) = thetaLower.t();
                upperSample.row(sampleIndex) = thetaUpper.t();
            }

            if (pSampleThinning > 0 && (index % pSampleThinning == 0)) {
                unsigned int sampleIndex = index / pSampleThinning;
                pSample.row(sampleIndex) = pCurrent.t();
            }

            if (zSampleThinning > 0 && (index % zSampleThinning == 0)) {
                zSample.row(index / zSampleThinning) = zCurrent.t();
            }

            if (yMissingSampleThinning > 0 && (index % yMissingSampleThinning == 0)) {
                yMissingSample.row(index / yMissingSampleThinning) = y(yMissingIndices).t();
            }
        }

        if (progress) {
            ++progressBar;
        }
    }

    List sample;
    if (distributionSampleThinning > 0) {
        sample["lower"] = wrap(lowerSample);
        sample["upper"] = wrap(upperSample);
    }
    if (pSampleThinning > 0) {
        sample["p"] = wrap(pSample);
    }
    if (zSampleThinning > 0) {
        sample["z"] = wrap(zSample);
    }
    if (yMissingSampleThinning > 0) {
        sample["y_missing"] = wrap(yMissingSample);
    }
    List results;
    results["sample"] = sample;

    return results;
}
