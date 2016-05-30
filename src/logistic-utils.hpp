#ifndef SRC_LOGISTIC_UTILS_HPP_
#define SRC_LOGISTIC_UTILS_HPP_

#include <RcppArmadillo.h>

double logisticLogLikelihood(const arma::mat delta, const arma::ucolvec z, const arma::mat explanatoryVariables);
arma::mat logisticHessian(const arma::mat p, const arma::mat explanatoryVariables);
arma::colvec logisticGrad(const arma::mat p, const arma::ucolvec z, const arma::mat explanatoryVariables);

arma::mat getLogisticP(const arma::colvec delta, const arma::mat explanatoryVariables);
arma::mat getLogisticP(const arma::mat delta, const arma::mat explanatoryVariables);

#endif  // SRC_LOGISTIC_UTILS_HPP_
