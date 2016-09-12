#include <algorithm>
#include <RcppArmadillo.h>

#include "gev.hpp"

using Rcpp::CharacterVector;
using Rcpp::NumericVector;

double rgev(double mu, double sigma, double xi) {
    return mu - (sigma / xi) * (1 - pow(-log(R::runif(0, 1)), -xi));
}

//' @export
// [[Rcpp::export(name="rgev")]]
NumericVector rgevVector(int n, double mu, double sigma, double xi) {
    NumericVector output(n);
    for (int i = 0; i < n; ++i) {
        output[i] = rgev(mu, sigma, xi);
    }
    return output;
}

double dgev(double x, double mu, double sigma, double xi, bool returnLog) {
    if (xi == 0) {
        double t = (x - mu) / sigma;

        if (returnLog) {
            return log(1 / sigma) - t - exp(-t);
        } else {
            return (1 / sigma) * exp(-t) * exp(-exp(-t));
        }
    } else {
        double t = 1 + (xi / sigma) * (x - mu);
        double powt = pow(t, -1 / xi);
        if (returnLog) {
            return -log(sigma) - (1 / xi + 1) * log(t) - powt;
        } else {
            return powt * exp(-powt) / (sigma * t);
        }
    }
}

// [[Rcpp::export(name=".estimatePwm")]]
double estimatePwm(NumericVector x, int r) {
    double sum = 0;
    int n = x.length();
    std::sort(x.begin(), x.end());
    // sum((i - 1) * ... * (i - r) * x)
    for (int i = r; i < n; ++i) {
        double value = x[i];
        for (int j = 1; j <= r; ++j) {
            value *= 1 + i - j;
        }
        sum += value;
    }
    // sum / (n * (n - 1) * ... * (n - r))
    for (int j = 0; j <= r; ++j) {
        sum /= (n - j);
    }

    return sum;
}

//' @export
// [[Rcpp::export(name="gevPwmEstimate")]]
NumericVector gevPwmEstimate(NumericVector x) {
    double M0 = estimatePwm(x, 0);
    double M1 = estimatePwm(x, 1);
    double M2 = estimatePwm(x, 2);

    double c = (2 * M1 - M0) / (3 * M2 - M0) - (log(2) / log(3));
    double xi = -7.8590 * c - 2.9554 * pow(c, 2);
    double sigma = (xi * (2 * M1 - M0)) / (tgamma(1 - xi) * (pow(2, xi) - 1));
    double mu = M0 + (sigma / xi) * (1 - tgamma(1 - xi));

    NumericVector params(3);
    params[0] = mu;
    params[1] = sigma;
    params[2] = xi;
    params.names() = CharacterVector::create("mu", "sigma", "xi");

    return params;
}


//' @export
// [[Rcpp::export(name="gevPwmEstimateConstrained")]]
NumericVector gevPwmEstimateConstrained(NumericVector x, double supportLim) {
    double M0 = estimatePwm(x, 0);
    double M1 = estimatePwm(x, 1);

    double xi = log((2 * M1 - supportLim) / (M0 - supportLim)) / log(2);
    double sigma = xi * (M0 - supportLim) / tgamma(1 - xi);
    double mu = supportLim + sigma / xi;

    NumericVector params(3);
    params[0] = mu;
    params[1] = sigma;
    params[2] = xi;
    params.names() = CharacterVector::create("mu", "sigma", "xi");

    return params;
}
