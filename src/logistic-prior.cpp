#include <RcppArmadillo.h>
#include "logistic-prior.hpp"

using arma::colvec;
using arma::mat;

using Rcpp::as;
using Rcpp::List;

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
