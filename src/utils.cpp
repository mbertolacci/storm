#include <vector>
#include <RcppArmadillo.h>
#include "utils.hpp"

using arma::colvec;
using arma::mat;
using arma::max;
using arma::ucolvec;

MissingValuesPair findMissingValues(Rcpp::NumericVector y) {
    int n = y.length();
    ucolvec yMissingIndices(n);
    std::vector<bool> yIsMissing(n);
    int nMissing = 0;
    for (int i = 0; i < n; ++i) {
        if (Rcpp::NumericVector::is_na(y[i])) {
            yMissingIndices[nMissing] = i;
            yIsMissing[i] = true;
            ++nMissing;
        } else {
            yIsMissing[i] = false;
        }
    }
    yMissingIndices.resize(nMissing);

    return std::make_pair(yMissingIndices, yIsMissing);
}

void sampleMissingY(
    colvec &y, colvec &logY, const ucolvec yMissingIndices, const ucolvec zCurrent,
    const ParameterBoundDistribution boundLowerDistribution, const ParameterBoundDistribution boundUpperDistribution
) {
    for (unsigned int i = 0; i < yMissingIndices.n_elem; ++i) {
        int index = yMissingIndices[i];
        if (zCurrent[index] == 1) {
            y[index] = 0;
        } else if (zCurrent[index] == 2) {
            y[index] = boundLowerDistribution.sample();
        } else {
            y[index] = boundUpperDistribution.sample();
        }
        logY[index] = log(y[index]);
    }
}

int sampleSingleZ(double pLower, double pUpper) {
    double u = R::runif(0, 1);
    if (u < pLower) {
        return 2;
    } else if (u < pLower + pUpper) {
        return 3;
    } else {
        return 1;
    }
}

ucolvec sampleZ(
    const mat pCurrent,
    const colvec y, const std::vector<bool> yIsMissing,
    const ParameterBoundDistribution boundLowerDistribution, const ParameterBoundDistribution boundUpperDistribution
) {
    ucolvec z(y.n_elem);

    for (unsigned int i = 0; i < y.n_elem; ++i) {
        if (yIsMissing[i]) {
            z[i] = sampleSingleZ(pCurrent(i, 0), pCurrent(i, 1));
        } else {
            if (y[i] == 0) {
                z[i] = 1;
            } else {
                double p2 = 0;
                double p3 = 0;
                if (boundLowerDistribution.isInSupport(y[i])) {
                    p2 = pCurrent(i, 0) * boundLowerDistribution.pdf(y[i]);
                }
                if (boundUpperDistribution.isInSupport(y[i])) {
                    p3 = pCurrent(i, 1) * boundUpperDistribution.pdf(y[i]);
                }

                double u = R::runif(0, p2 + p3);
                if (u < p2) {
                    z[i] = 2;
                } else {
                    z[i] = 3;
                }
            }
        }
    }

    return z;
}
