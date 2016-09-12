#ifndef SRC_LOGISTIC_MLE_HPP_
#define SRC_LOGISTIC_MLE_HPP_

#include "logistic-prior.hpp"

typedef struct {
    arma::mat delta;
    arma::mat hessian;
} LogisticMLEResult;

LogisticMLEResult logisticMaximumLikelihoodEstimate(
    const arma::mat deltaStart, const arma::ucolvec z, const arma::mat explanatoryVariables,
    const LogisticParameterPrior prior,
    double precision = 0.001, unsigned int maxIterations = 50
);

#endif  // SRC_LOGISTIC_MLE_HPP_
