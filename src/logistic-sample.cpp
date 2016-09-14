#include <chrono>
#include <RcppArmadillo.h>
#include <vector>

#include "hypercube4.hpp"
#include "logistic-sampler.hpp"
#include "progress.hpp"
#include "rng.hpp"
#include "utils.hpp"

using arma::colvec;
using arma::cube;
using arma::field;
using arma::mat;
using arma::umat;

using Rcpp::as;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;
using Rcpp::Rcout;
using Rcpp::StringVector;

using ptsm::RNG;

const unsigned int CHECK_INTERRUPT_INTERVAL = 10;

// [[Rcpp::export(name=".ptsm_logistic_sample")]]
List logisticSample(
    unsigned int nSamples, unsigned int burnIn,
    List panelY, List panelDesignMatrix, unsigned int order,
    StringVector distributionNames, List priors, List samplingSchemes,
    List panelZStart, IntegerVector panelZ0Start, List thetaStart,
    List panelDeltaStart, NumericVector deltaFamilyMeanStart, NumericMatrix deltaFamilyVarianceStart,
    Rcpp::Nullable<NumericMatrix> deltaDesignMatrix,
    List thinning,
    unsigned int verbose = 0, bool progress = false
) {
    RNG::initialise();

    LogisticSampler sampler(
        panelY, panelDesignMatrix, order,
        distributionNames, priors, samplingSchemes,
        panelZStart, panelZ0Start, thetaStart,
        panelDeltaStart, deltaFamilyMeanStart, deltaFamilyVarianceStart,
        deltaDesignMatrix
    );

    return LogisticSample<LogisticSampler>(
        sampler,
        nSamples, burnIn,
        thinning["distribution"], thinning["delta"], thinning["family"],
        thinning["z0"], thinning["z"], thinning["y_missing"],
        progress, CHECK_INTERRUPT_INTERVAL
    ).asList();
}



// [[Rcpp::export(name=".ptsm_logistic_sample_y")]]
List logisticSampleY(
    List panelExplanatoryVariablesR, List panelDeltaSampleR, List panelZ0SampleR,
    List distributionSampleR,
    StringVector distributionNames
) {
    RNG::initialise();

    field<mat> distributionSample(distributionNames.length());
    std::vector<Distribution> distributions;
    for (unsigned int k = 0; k < distributionNames.length(); ++k) {
        distributionSample[k] = trans(as<mat>(distributionSampleR[k]));
        distributions.push_back(Distribution(distributionNames[k]));
    }

    unsigned int nLevels = panelExplanatoryVariablesR.length();
    unsigned int nComponents = distributions.size();

    field<mat> panelExplanatoryVariables = fieldFromList<mat>(panelExplanatoryVariablesR);
    for (unsigned int level = 0; level < nLevels; ++level) {
        // Add room for the latent variable lags
        panelExplanatoryVariables[level].resize(
            panelExplanatoryVariables[level].n_rows,
            panelExplanatoryVariables[level].n_cols + nComponents
        );
    }

    unsigned int nSamples = distributionSample[0].n_cols;
    unsigned int nDeltas = panelExplanatoryVariables[0].n_cols;

    field<cube> panelDeltaSample(panelDeltaSampleR.length());
    mat panelZ0Sample(nSamples, panelDeltaSampleR.length());
    for (unsigned int level = 0; level < panelDeltaSampleR.length(); ++level) {
        panelDeltaSample[level] = as<cube>(panelDeltaSampleR[level]);
        panelZ0Sample.col(level) = as<colvec>(panelZ0SampleR[level]);
    }

    // Output
    field<mat> panelYSample(nLevels);
    field<umat> panelYSampleZ(nLevels);
    for (unsigned int level = 0; level < nLevels; ++level) {
        panelYSample[level] = mat(nSamples, panelExplanatoryVariables[level].n_rows);
        panelYSampleZ[level] = umat(nSamples, panelExplanatoryVariables[level].n_rows);
    }

    ProgressBar progressBar(nSamples);
    for (unsigned int sampleIndex = 0; sampleIndex < nSamples; ++sampleIndex) {
        #pragma omp parallel for
        for (unsigned int level = 0; level < nLevels; ++level) {
            mat delta = panelDeltaSample[level].slice(sampleIndex);
            unsigned int previousZ = panelZ0Sample(sampleIndex, level);

            colvec componentSums(nComponents + 1, arma::fill::zeros);
            colvec componentExpDiff(nComponents + 1);
            colvec p(nComponents);

            for (unsigned int i = 0; i < panelYSample[level].n_cols; ++i) {
                for (unsigned int k = 0; k < nComponents; ++k) {
                    panelExplanatoryVariables[level](i, nDeltas + k - nComponents) = previousZ == k + 2;
                }

                for (unsigned int k = 0; k < nComponents; ++k) {
                    componentSums[k] = arma::dot(delta.row(k), panelExplanatoryVariables[level].row(i));
                }
                componentExpDiff = exp(componentSums - arma::max(componentSums));
                p = componentExpDiff.head(nComponents) / arma::sum(componentExpDiff);

                unsigned int z = sampleSingleZ(p);
                double y = 0;

                if (z > 1) {
                    y = distributions[z - 2].sample(distributionSample[z - 2].col(sampleIndex));
                }

                panelYSample[level](sampleIndex, i) = y;
                panelYSampleZ[level](sampleIndex, i) = z;

                previousZ = z;
            }
        }

        Rcpp::checkUserInterrupt();
        ++progressBar;
    }

    List results;
    results["panel_y_sample"] = listFromField(panelYSample);
    results["panel_y_sample_z"] = listFromField(panelYSampleZ);

    return results;
}
