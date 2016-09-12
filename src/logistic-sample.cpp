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
    NumericMatrix thetaLowerSampleR, NumericMatrix thetaUpperSampleR,
    StringVector distributionNames
) {
    RNG::initialise();

    mat thetaLowerSample = trans(as<mat>(thetaLowerSampleR));
    mat thetaUpperSample = trans(as<mat>(thetaUpperSampleR));
    Distribution lowerDistribution(distributionNames[0]);
    Distribution upperDistribution(distributionNames[1]);

    unsigned int nLevels = panelExplanatoryVariablesR.length();

    field<mat> panelExplanatoryVariables = fieldFromList<mat>(panelExplanatoryVariablesR);
    for (unsigned int level = 0; level < nLevels; ++level) {
        // Add room for the latent variable lags
        panelExplanatoryVariables[level].resize(
            panelExplanatoryVariables[level].n_rows,
            panelExplanatoryVariables[level].n_cols + 2
        );
    }

    unsigned int nSamples = thetaLowerSample.n_cols;
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

            for (unsigned int i = 0; i < panelYSample[level].n_cols; ++i) {
                panelExplanatoryVariables[level](i, nDeltas - 2) = previousZ == 2;
                panelExplanatoryVariables[level](i, nDeltas - 1) = previousZ == 3;
                double lowerSum = arma::dot(delta.row(0), panelExplanatoryVariables[level].row(i));
                double upperSum = arma::dot(delta.row(1), panelExplanatoryVariables[level].row(i));

                double expDiff = exp(upperSum - lowerSum);
                double pLower = 1 / (1 + exp(-lowerSum) + expDiff);
                double pUpper = 1 / (1 + exp(-upperSum) + 1 / expDiff);

                unsigned int z = sampleSingleZ({ pLower, pUpper });
                double y = 0;

                if (z == 2) {
                    y = lowerDistribution.sample(thetaLowerSample.col(sampleIndex));
                } else if (z == 3) {
                    y = upperDistribution.sample(thetaUpperSample.col(sampleIndex));
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
