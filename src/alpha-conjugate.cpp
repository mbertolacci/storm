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

    // If you're not satisfied with five Newton-Raphson iterations you're just being unreasonable
    x = x - (R::digamma(x) - y) / R::trigamma(x);
    x = x - (R::digamma(x) - y) / R::trigamma(x);
    x = x - (R::digamma(x) - y) / R::trigamma(x);
    x = x - (R::digamma(x) - y) / R::trigamma(x);
    x = x - (R::digamma(x) - y) / R::trigamma(x);

    return x;
}

double logDensity(double alpha, double logBeta, double logP, double q, double r) {
    return (alpha - 1) * logP - alpha * q * logBeta - r * lgamma(alpha);
}

double dLogDensity(double alpha, double logBeta, double logP, double q, double r) {
    return logP - q * logBeta - r * R::digamma(alpha);
}

double rGammaShapeConjugate(double beta, double logP, double q, double r) {
    // This is like Adaptive Rejection Sampling, but with just two fixed points
    // x1 and x2 to form a hull. The hull stretches from x = 0, to the intersection
    // between the tangent lines, and then to +Inf

    // Notationally, h = logDensity

    double logBeta = log(beta);
    double mode = idigamma((logP - q * logBeta) / r);

    // Pick as our two points +- two standard deviations (based on MLE variance approximation)
    double sdEstimate = 1 / sqrt(r * R::trigamma(mode));
    // Must be > 0
    double x1 = std::max(mode / 2, mode - 2 * sdEstimate);
    double x2 = mode + 2 * sdEstimate;

    double hX1 = logDensity(x1, logBeta, logP, q, r);
    double slopeHX1 = dLogDensity(x1, logBeta, logP, q, r);

    double hX2 = logDensity(x2, logBeta, logP, q, r);
    double slopeHX2 = dLogDensity(x2, logBeta, logP, q, r);

    // H value of the infimum, which is x = 0
    double hInf = hX1 - x1 * slopeHX1;

    // Point of intersection between the two tangent lines, and H value for that
    double intersection = x1 + (hX1 - hX2 + slopeHX2 * (x2 - x1)) / (slopeHX2 - slopeHX1);
    double hIntersection = slopeHX1 * (intersection - x1) + hX1;

    double maxH = std::max(hInf, hIntersection);

    double expHInf = exp(hInf - maxH);
    double expHIntersection = exp(hIntersection - maxH);

    double sIntersection = (expHIntersection - expHInf) / slopeHX1;
    double sSup = -expHIntersection / slopeHX2;
    double sTotal = sIntersection + sSup;

    for (unsigned int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        // Draw x from the convex hull
        double u = rng.randu();
        double x;
        double hX;
        if (u < sIntersection / sTotal) {
            x = (
                log(expHInf + slopeHX1 * sTotal * u)
                - hInf + maxH
            ) / slopeHX1;
            hX = hInf + x * slopeHX1;
        } else {
            x = intersection + (
                log(expHIntersection + slopeHX2 * (sTotal * u - sIntersection))
                - hIntersection + maxH
            ) / slopeHX2;
            hX = hIntersection + (x - intersection) * slopeHX2;
        }

        // Accept/reject
        double v = log(rng.randu());
        if (v < logDensity(x, logBeta, logP, q, r) - hX) {
            return x;
        }
    }

    Rcpp::stop("Did not finish in time");
    return 0;
}

// [[Rcpp::export(name="rgammashapeconjugate")]]
NumericVector rGammaShapeConjugateR(
    unsigned int n, double beta,
    double logP = NA_REAL, double q = NA_REAL, double r = NA_REAL,
    Rcpp::Nullable<NumericVector> x = R_NilValue, Rcpp::Nullable<NumericVector> prior = R_NilValue
) {
    RNG::initialise();

    if (!x.isNull()) {
        NumericVector xInner = x.get();

        logP = Rcpp::sum(Rcpp::log(xInner));
        q = r = xInner.length();

        if (!prior.isNull()) {
            NumericVector priorInner = prior.get();

            if (priorInner.length() < 3) {
                Rcpp::stop("Prior must contain three elements");
            }

            logP += priorInner[0];
            q += priorInner[1];
            r += priorInner[2];
        }
    }

    NumericVector results(n);
    for (unsigned int i = 0; i < n; ++i) {
        results[i] = rGammaShapeConjugate(beta, logP, q, r);
    }

    return results;
}
