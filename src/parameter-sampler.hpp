#ifndef SRC_PARAMETER_SAMPLER_HPP_
#define SRC_PARAMETER_SAMPLER_HPP_

#include <RcppArmadillo.h>

#include "distribution.hpp"

class ParameterSampler {
 public:
    ParameterSampler() = default;
    ParameterSampler(Rcpp::List prior, Rcpp::List samplingScheme, Distribution distribution);

    void printAcceptanceRatio(int nIterations);
    void resetAcceptCount();

    arma::colvec sample(const arma::colvec currentParameters, const DataBoundDistribution &boundDistribution);

 private:
    bool useMle_;
    bool useObservedInformation_;
    bool ignoreCovariance_;
    int accept_;
    double observedInformationInflationFactor_;

    arma::mat uniformPriorBounds_;
    arma::mat covarianceCholesky_;

    bool satisfiesPrior(arma::colvec parameters) const;
};

#endif  // SRC_PARAMETER_SAMPLER_HPP_
