#include <RcppArmadillo.h>
#include <vector>

#include "distribution.hpp"
#include "parameter-sampler.hpp"
#include "rng.hpp"
#include "utils.hpp"
#include "progress.hpp"

#include "independent-sampler.hpp"

using arma::colvec;
using arma::field;
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

std::vector<ParameterBoundDistribution> getParameterBoundDistributions(
    std::vector<Distribution> distributions,
    field<colvec> parameters
) {
    std::vector<ParameterBoundDistribution> output;
    for (unsigned int k = 0; k < distributions.size(); ++k) {
        output.push_back(ParameterBoundDistribution(
            parameters[k],
            distributions[k]
        ));
    }
    return output;
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
    unsigned int nComponents = distributionNames.length();

    // Priors and samplers
    std::vector<Distribution> distributions;
    std::vector<ParameterSampler> distributionSamplers;

    List distributionsPrior = priors["distributions"];
    List distributionsSamplingSchemes = samplingSchemes["distributions"];
    for (unsigned int k = 0; k < nComponents; ++k) {
        distributions.push_back(Distribution(distributionNames[k]));
        distributionSamplers.push_back(ParameterSampler(
            distributionsPrior[k], distributionsSamplingSchemes[k],
            distributions[k]
        ));
    }
    colvec pPrior = as<colvec>(priors["p"]);

    // Data and missing data
    colvec y = as<colvec>(yR);
    colvec logY = log(y);

    MissingValuesPair missingValuesPair = findMissingValues(yR);
    ucolvec yMissingIndices = missingValuesPair.first;
    std::vector<bool> yIsMissing = missingValuesPair.second;

    // Starting values for parameters
    ucolvec zCurrent = as<ucolvec>(zStart);
    field<colvec> distributionCurrent = fieldFromList<colvec>(distributionsStart);

    colvec pCurrent(nComponents + 1);
    mat pCurrentGivenZ(y.n_elem, nComponents);

    // Samples for output
    field<mat> distributionSample(nComponents);
    if (distributionSampleThinning > 0) {
        unsigned int nDistributionSamples = ceil(
            static_cast<double>(nSamples) / static_cast<double>(distributionSampleThinning)
        );
        for (unsigned int k = 0; k < nComponents; ++k) {
            distributionSample[k].set_size(nDistributionSamples, distributionCurrent[k].n_elem);
        }
    }
    mat pSample;
    if (pSampleThinning > 0) {
        unsigned int nPSamples = ceil(
            static_cast<double>(nSamples) / static_cast<double>(pSampleThinning)
        );
        pSample = mat(nPSamples, nComponents + 1);
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
        y, logY, yMissingIndices, zCurrent, getParameterBoundDistributions(distributions, distributionCurrent)
    );

    ProgressBar progressBar(nIterations);
    for (unsigned int iteration = 0; iteration < nIterations; ++iteration) {
        // Sample p
        pCurrent = sampleP(zCurrent, pPrior);

        // Sample distribution parameters
        for (unsigned int k = 0; k < distributions.size(); ++k) {
            distributionCurrent[k] = distributionSamplers[k].sample(
                distributionCurrent[k],
                DataBoundDistribution(y, logY, zCurrent, k + 2, distributions[k])
            );
        }

        // Sample \bm{z}
        for (unsigned int k = 0; k < distributions.size(); ++k) {
            pCurrentGivenZ.col(k).fill(pCurrent[k + 1]);
        }
        zCurrent = sampleZ(
            pCurrentGivenZ, y, yIsMissing,
            getParameterBoundDistributions(distributions, distributionCurrent)
        );

        // Sample missing values \bm{y^*}
        sampleMissingY(
            y, logY, yMissingIndices, zCurrent, getParameterBoundDistributions(distributions, distributionCurrent)
        );

        if (iteration >= burnIn) {
            int index = iteration - burnIn;

            if (distributionSampleThinning > 0 && (index % distributionSampleThinning == 0)) {
                unsigned int sampleIndex = index / distributionSampleThinning;
                for (unsigned int k = 0; k < nComponents; ++k) {
                    distributionSample[k].row(sampleIndex) = distributionCurrent[k].t();
                }
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
        sample["distribution"] = listFromField(distributionSample);
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
