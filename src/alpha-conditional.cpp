#include <RcppArmadillo.h>
#include "rng.hpp"

using Rcpp::NumericVector;
using ptsm::RNG;
using ptsm::rng;

const double TOLERANCE = sqrt(std::numeric_limits<double>::epsilon());
const unsigned int MAX_ITERATIONS = 10000;

// Inverse digamma function
// [[Rcpp::export]]
double idigamma(double y) {
    double x;

    // As per Minka (2000)
    if (y > -2.22) {
        x = exp(y) + 0.5;
    } else {
        x = -1 / (y + R::digamma(1));
    }

    // If you're not satisfied with four Newton-Raphson iterations you're just being unreasonable
    x = x - (R::digamma(x) - y) / R::trigamma(x);
    x = x - (R::digamma(x) - y) / R::trigamma(x);
    x = x - (R::digamma(x) - y) / R::trigamma(x);
    x = x - (R::digamma(x) - y) / R::trigamma(x);

    return x;
}

double logDensity(double alpha, double logBeta, double logP, double q, double r) {
    return (alpha - 1) * logP - alpha * q * logBeta - r * lgamma(alpha);
}

double rGammaShapeConjugate(double beta, double logP, double q, double r) {
    double logBeta = log(beta);
    double mode = idigamma((logP - q * logBeta) / r);
    double logFMode = logDensity(mode, logBeta, logP, q, r);

    double X;

    for (unsigned int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        double T;
        double u = rng.randu();

        if (u <= 4.0 / 11.0) {
            X = rng.randu() / 2;
            T = 2 * rng.randu();
        } else if (u <= 7.0 / 11.0) {
            X = 0.5 * (1 + std::min(rng.randu(), 2 * rng.randu()));
            T = rng.randu() * (1 + 2 * (1 - X));
        } else {
            double u2 = rng.randu();
            double u3 = rng.randu();
            X = 1 - log(u3);
            T = u2 * u3;
        }

        double fX = exp(logDensity(mode + X, logBeta, logP, q, r) - logFMode);
        double fMinusX = exp(logDensity(mode - X, logBeta, logP, q, r) - logFMode);

        if (T <= fX + fMinusX) {
            if (rng.randu() > fX / (fX + fMinusX)) {
                X = -X;
            }

            if (mode + X < 0) {
                // Try again
                continue;
            }

            return mode + X;
        }
    }

    Rcpp::stop("Did not finish in time");
    return 0;
}

// [[Rcpp::export(name="rgammashapeconjugate")]]
NumericVector rGammaShapeConjugateR(
    unsigned int n, double beta, NumericVector x, Rcpp::Nullable<NumericVector> prior = R_NilValue
) {
    RNG::initialise();

    NumericVector priorInner;
    if (prior.isNull()) {
        priorInner = NumericVector(3);
        priorInner[0] = 0;
        priorInner[1] = 0;
        priorInner[2] = 0;
    } else {
        priorInner = prior.get();
    }

    if (priorInner.length() < 3) {
        Rcpp::stop("Prior must contain three elements");
    }

    double sumLogX = Rcpp::sum(Rcpp::log(x));
    double nX = x.length();

    NumericVector results(n);
    for (unsigned int i = 0; i < n; ++i) {
        results[i] = rGammaShapeConjugate(
            beta, priorInner[0] + sumLogX, priorInner[1] + nX, priorInner[2] + nX
        );
    }
    return results;
}
