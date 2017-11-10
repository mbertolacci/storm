#ifndef SRC_PARAMETER_SAMPLER_HPP_
#define SRC_PARAMETER_SAMPLER_HPP_

#include <RcppArmadillo.h>

#include "distribution.hpp"

class ParameterSampler {
 public:
    ParameterSampler() = default;
    ParameterSampler(Rcpp::List prior, Rcpp::List samplingScheme, Distribution distribution);

    arma::colvec sample(const arma::colvec currentParameters, const DataBoundDistribution &boundDistribution);

 private:
    bool useGibbs_;

    bool useMle_;
    bool useObservedInformation_;
    double observedInformationInflationFactor_;

    // Gamma distribution priors
    arma::colvec priorAlpha_;
    arma::colvec priorBeta_;
    // Log normal distribution priors
    arma::colvec priorMu_;
    arma::colvec priorTau_;

    arma::mat uniformPriorBounds_;
    arma::mat covarianceCholesky_;

    arma::colvec sampleGibbs_(
        const arma::colvec currentParameters, const DataBoundDistribution& boundDistribution
    );

    arma::colvec sampleMetropolisHastings_(
        const arma::colvec currentParameters, const DataBoundDistribution& boundDistribution
    );

    bool satisfiesPrior_(arma::colvec parameters) const;
};

#endif  // SRC_PARAMETER_SAMPLER_HPP_
