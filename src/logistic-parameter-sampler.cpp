#include <algorithm>

#include <RcppArmadillo.h>

#include "logistic-utils.hpp"
#include "logistic-parameter-sampler.hpp"

using arma::colvec;
using arma::mat;
using arma::max;
using arma::ucolvec;

using Rcpp::as;
using Rcpp::List;
using Rcpp::Rcout;

LogisticParameterPrior::LogisticParameterPrior(List prior)
    : isUniform_(false) {
    if (strcmp(prior["type"], "uniform") == 0) {
        isUniform_ = true;
        uniformPriorBounds_ = as<mat>(prior["bounds"]);
    }
}

bool LogisticParameterPrior::isInSupport(const mat delta) const {
    if (!isUniform_) return true;

    for (unsigned int i = 0; i < delta.n_cols; ++i) {
        if (delta(0, i) < uniformPriorBounds_(i, 0)) return false;
        if (delta(0, i) > uniformPriorBounds_(i, 1)) return false;

        if (delta(1, i) < uniformPriorBounds_(i + delta.n_cols, 0)) return false;
        if (delta(1, i) > uniformPriorBounds_(i + delta.n_cols, 1)) return false;
    }
    return true;
}

double LogisticParameterPrior::logPdf(const mat delta, const mat means, const mat variances) const {
    if (isUniform_) return 0;

    double sum = 0;

    for (unsigned int i = 0; i < delta.n_elem; ++i) {
        sum += -0.5 * (delta[i] * delta[i] - 2 * delta[i] * means[i]) / variances[i];
    }

    return sum;
}

double LogisticParameterPrior::logPdf(const mat delta) const {
    return logPdf(delta, means_, variances_);
}

colvec LogisticParameterPrior::grad(const colvec delta, const mat means, const mat variances) const {
    unsigned int nDeltas = delta.n_elem / 2;

    colvec grad(size(delta));
    for (unsigned int j = 0; j < nDeltas; ++j) {
        grad[j] = -(delta[j] - means(0, j)) / variances(0, j);
        grad[nDeltas + j] = -(delta[nDeltas + j] - means(1, j)) / variances(1, j);
    }
    return grad;
}

colvec LogisticParameterPrior::grad(const colvec delta) const {
    return grad(delta, means_, variances_);
}

colvec LogisticParameterPrior::grad(const mat delta, const mat means, const mat variances) const {
    return grad((colvec) vectorise(delta, 1).t(), means, variances);
}

colvec LogisticParameterPrior::grad(const mat delta) const {
    return grad(delta, means_, variances_);
}

mat LogisticParameterPrior::hessian(const colvec delta, const mat means, const mat variances) const {
    unsigned int nDeltas = delta.n_elem / 2;

    mat hessian(2 * nDeltas, 2 * nDeltas, arma::fill::zeros);
    // Load up the diagonals
    for (unsigned int j = 0; j < nDeltas; ++j) {
        hessian(j, j) = -1 / variances(0, j);
        hessian(nDeltas + j, nDeltas + j) = -1 / variances(1, j);
    }
    return hessian;
}

mat LogisticParameterPrior::hessian(const colvec delta) const {
    return hessian(delta, means_, variances_);
}

mat LogisticParameterPrior::hessian(const mat delta, const mat means, const mat variances) const {
    return hessian((colvec) vectorise(delta, 1).t(), means, variances);
}

mat LogisticParameterPrior::hessian(const mat delta) const {
    return hessian(delta, means_, variances_);
}

typedef struct {
    mat delta;
    mat hessian;
} LogisticMLEResult;

LogisticMLEResult logisticMaximumLikelihoodEstimate(
    const mat deltaStart, const ucolvec z, const mat explanatoryVariables,
    const LogisticParameterPrior prior,
    double precision = 0.001, unsigned int maxIterations = 50
) {
    // Stack the rows into one column
    // colvec deltaCurrent = vectorise(deltaStart, 1).t();
    colvec deltaCurrent(2 * explanatoryVariables.n_cols, arma::fill::zeros);
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

LogisticParameterSampler::LogisticParameterSampler(List samplingScheme)
    : accept_(0) {
    if (samplingScheme.containsElementNamed("observed_information_inflation_factor")) {
        observedInformationInflationFactor_ = samplingScheme["observed_information_inflation_factor"];
    } else {
        observedInformationInflationFactor_ = 1.0;
    }

    if (samplingScheme.containsElementNamed("ignore.covariance")) {
        ignoreCovariance_ = static_cast<int>(samplingScheme["ignore.covariance"]);
    } else {
        ignoreCovariance_ = false;
    }
}

void LogisticParameterSampler::printAcceptanceRatios(int nIterations) const {
    printAcceptanceRatios(nIterations, 1);
}

void LogisticParameterSampler::printAcceptanceRatios(int nIterations, int factor) const {
    Rcout << "Delta acceptance " << (static_cast<double>(accept_) / static_cast<double>(nIterations * factor)) << "\n";
}

void LogisticParameterSampler::resetAcceptCounts() {
    accept_ = 0;
}

mat LogisticParameterSampler::sample(
    const mat deltaCurrent, const mat pCurrent, const ucolvec zCurrent, const mat explanatoryVariables,
    const LogisticParameterPrior prior
) {
    unsigned int nDeltas = deltaCurrent.n_cols;

    mat deltaMean(deltaCurrent);
    mat deltaProposal(deltaCurrent);

    LogisticMLEResult mleResult = logisticMaximumLikelihoodEstimate(
        deltaCurrent, zCurrent, explanatoryVariables, prior
    );
    deltaMean = mleResult.delta;
    mat hessian = mleResult.hessian;

    mat cholesky = chol(observedInformationInflationFactor_ * (-hessian).i());

    colvec unitNormals(2 * nDeltas);
    colvec moves(2 * nDeltas);

    while (true) {
        for (unsigned int i = 0; i < unitNormals.n_elem; ++i) {
            unitNormals[i] = R::rnorm(0, 1);
        }

        moves = cholesky * unitNormals;

        for (unsigned int i = 0; i < nDeltas; ++i) {
            deltaProposal(0, i) = deltaMean(0, i) + moves[i];
            deltaProposal(1, i) = deltaMean(1, i) + moves[nDeltas + i];
        }

        if (prior.isInSupport(deltaProposal)) {
            break;
        }
    }

    double currentLogDensity = logisticLogLikelihood(
        deltaCurrent, zCurrent, explanatoryVariables
    ) + prior.logPdf(deltaCurrent);

    double proposalLogDensity = logisticLogLikelihood(
        deltaProposal, zCurrent, explanatoryVariables
    ) + prior.logPdf(deltaProposal);

    double alpha = proposalLogDensity - currentLogDensity;
    if (log(R::runif(0, 1)) < alpha) {
        accept_++;
        return deltaProposal;
    }

    return deltaCurrent;
}
