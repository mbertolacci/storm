#include <RcppArmadillo.h>
#include <utility>
#include "logistic-sampler-mpi.hpp"
#include "progress.hpp"
#include "rng.hpp"

using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;
using Rcpp::Rcout;
using Rcpp::StringVector;
using Rcpp::checkUserInterrupt;
using Rcpp::wrap;

using ptsm::RNG;
using ptsm::rng;

const unsigned int CHECK_INTERRUPT_INTERVAL = 10;

// [[Rcpp::export(name=".logistic_sample_mpi")]]
List logisticSampleMPI(
    unsigned int nSamples, unsigned int burnIn,
    List panelY, List panelDesignMatrix, unsigned int order,
    StringVector distributionNames, List priors, List samplingSchemes,
    List thetaStart,
    List panelDeltaStart, NumericVector deltaFamilyMeanStart, NumericMatrix deltaFamilyVarianceStart,
    Rcpp::Nullable<NumericMatrix> deltaDesignMatrix,
    List thinning,
    unsigned int verbose = 0, bool progress = false,
    int numThreads = 0
) {
    RNG::initialise();

    LogisticSamplerMPI sampler(
        panelY, panelDesignMatrix, order,
        distributionNames, priors, samplingSchemes,
        thetaStart,
        panelDeltaStart, deltaFamilyMeanStart, deltaFamilyVarianceStart,
        deltaDesignMatrix
    );

    return LogisticSample<LogisticSamplerMPI>(
        sampler,
        nSamples, burnIn,
        thinning["distribution"], thinning["delta"], thinning["family"],
        thinning["z0"], thinning["z"], thinning["y_missing"],
        progress, CHECK_INTERRUPT_INTERVAL
    ).asList();
}
