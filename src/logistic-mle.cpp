#include <algorithm>
#include <RcppArmadillo.h>

#include "logistic-mle.hpp"
#include "logistic-utils.hpp"

using Rcpp::Rcout;

using arma::colvec;
using arma::mat;
using arma::ucolvec;

LogisticMLEResult logisticMaximumLikelihoodEstimate(
    const mat deltaStart, const ucolvec z, const mat explanatoryVariables,
    const LogisticParameterPrior prior,
    double precision, unsigned int maxIterations
) {
    // Stack the rows into one column
    colvec deltaCurrent = vectorise(deltaStart, 1).t();
    colvec deltaPrevious(size(deltaCurrent));

    mat pCurrent(explanatoryVariables.n_rows, 2);
    colvec gradCurrent(2 * explanatoryVariables.n_cols);
    mat hessianCurrent(2 * explanatoryVariables.n_cols, 2 * explanatoryVariables.n_cols);

    unsigned int i;
    for (i = 0; i < maxIterations; ++i) {
        pCurrent = getLogisticP(deltaCurrent, explanatoryVariables);
        gradCurrent = logisticGrad(pCurrent, z, explanatoryVariables) + prior.grad(deltaCurrent);
        hessianCurrent = logisticHessian(pCurrent, explanatoryVariables) + prior.hessian(deltaCurrent);

        deltaPrevious = deltaCurrent;
        deltaCurrent -= hessianCurrent.i() * gradCurrent;

        if (max(abs(deltaCurrent - deltaPrevious)) < precision) {
            break;
        }
    }
    if (i == maxIterations) {
        Rcout << "WARNING: delta MLE did not converge\n";
        Rcout << deltaStart << "\n" << deltaCurrent << "\n";
        throw std::runtime_error("delta MLE did not converge");
    }

    LogisticMLEResult result;
    // Unstack the vector back into matrix form
    result.delta = mat(size(deltaStart));
    result.delta.row(0) = deltaCurrent.head(explanatoryVariables.n_cols).t();
    result.delta.row(1) = deltaCurrent.tail(explanatoryVariables.n_cols).t();
    result.hessian = hessianCurrent;

    return result;
}
