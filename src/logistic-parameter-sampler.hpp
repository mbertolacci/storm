#ifndef SRC_LOGISTIC_PARAMETER_SAMPLER_HPP_
#define SRC_LOGISTIC_PARAMETER_SAMPLER_HPP_

#include <RcppArmadillo.h>
#include "logistic-prior.hpp"

class LogisticParameterSampler {
 public:
    explicit LogisticParameterSampler(Rcpp::List samplingScheme);

    void printAcceptanceRatios(int nIterations) const;
    void printAcceptanceRatios(int nIterations, int factor) const;
    void resetAcceptCounts();

    arma::mat sample(
        const arma::mat deltaCurrent, const arma::mat pCurrent,
        const arma::ucolvec zCurrent, const arma::mat explanatoryVariables,
        const LogisticParameterPrior prior
    );

 private:
    int accept_;
    double observedInformationInflationFactor_;
    bool isPolson_;

    arma::mat samplePolson(
        const arma::mat deltaCurrent, const arma::mat pCurrent,
        const arma::ucolvec zCurrent, const arma::mat explanatoryVariables,
        const LogisticParameterPrior prior
    );

    arma::mat sampleMetropolisHastings(
        const arma::mat deltaCurrent, const arma::mat pCurrent,
        const arma::ucolvec zCurrent, const arma::mat explanatoryVariables,
        const LogisticParameterPrior prior
    );
};

#endif  // SRC_LOGISTIC_PARAMETER_SAMPLER_HPP_
