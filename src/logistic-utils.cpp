#include <chrono>
#include <RcppArmadillo.h>
#include "logistic-utils.hpp"

using arma::colvec;
using arma::distr_param;
using arma::mat;
using arma::randi;
using arma::ucolvec;

using Rcpp::Rcout;

mat getLogisticP(const colvec delta, const mat explanatoryVariables) {
    unsigned int nDeltas = explanatoryVariables.n_cols;
    mat p(explanatoryVariables.n_rows, 2);

    for (unsigned int i = 0; i < explanatoryVariables.n_rows; ++i) {
        double lowerSum = 0;
        double upperSum = 0;
        for (unsigned int j = 0; j < nDeltas; ++j) {
            lowerSum += delta[j] * explanatoryVariables.at(i, j);
            upperSum += delta[nDeltas + j] * explanatoryVariables.at(i, j);
        }

        double expDiff = exp(upperSum - lowerSum);
        p.at(i, 0) = 1 / (1 + exp(-lowerSum) + expDiff);
        p.at(i, 1) = 1 / (1 + exp(-upperSum) + 1 / expDiff);
    }

    return p;
}

mat getLogisticP(const mat delta, const mat explanatoryVariables) {
    mat p(explanatoryVariables.n_rows, delta.n_rows);

    // The extra element here always contains 0 in the last element
    colvec componentSums(delta.n_rows + 1, arma::fill::zeros);
    colvec componentExpDiff(delta.n_rows + 1);

    for (unsigned int i = 0; i < explanatoryVariables.n_rows; ++i) {
        for (unsigned int k = 0; k < delta.n_rows; ++k) {
            componentSums[k] = arma::dot(delta.row(k), explanatoryVariables.row(i));
        }
        componentExpDiff = exp(componentSums - arma::max(componentSums));
        double denominator = arma::sum(componentExpDiff);

        for (unsigned int k = 0; k < delta.n_rows; ++k) {
            p(i, k) = componentExpDiff[k] / denominator;
        }
    }

    return p;
}

mat logisticHessian(const mat p, const mat explanatoryVariables) {
    unsigned int nDeltas = explanatoryVariables.n_cols;

    // [
    //    \frac{\partial^2 \ell}{\partial \delta_1^2}        \frac{\partial^2 \ell}{\partial \delta_1 \delta_2}
    //    \frac{\partial^2 \ell}{\partial \delta_1 \delta_2} \frac{\partial^2 \ell}{\partial \delta_2^2}
    // ]
    mat hessian(2 * nDeltas, 2 * nDeltas, arma::fill::zeros);

    for (unsigned int j = 0; j < nDeltas; ++j) {
        for (unsigned int jj = j; jj < nDeltas; ++jj) {
            double topLeftTopRight = 0;
            double bottomRightTopRight = 0;
            double topRightTopRight = 0;

            // Inner loop without += or -= dependencies on j allow vectorisation
            for (unsigned int i = 0; i < explanatoryVariables.n_rows; ++i) {
                double p2 = p.at(i, 0);
                double p3 = p.at(i, 1);

                double xx = explanatoryVariables.at(i, j) * explanatoryVariables.at(i, jj);
                // Top left top right: delta(0, j) delta(0, jj)
                topLeftTopRight -= xx * p2 * (1 - p2);
                // Bottom right top right: delta(1, j) delta(1, jj)
                bottomRightTopRight -= xx * p3 * (1 - p3);
                // Top right top right: delta(0, j) delta(1, jj)
                topRightTopRight += xx * p2 * p3;
            }

            hessian.at(j, jj) = topLeftTopRight;
            hessian.at(nDeltas + j, nDeltas + jj) = bottomRightTopRight;
            hessian.at(j, nDeltas + jj) = topRightTopRight;
            hessian.at(jj, nDeltas + j) = topRightTopRight;
        }
    }

    return symmatu(hessian);
}

colvec logisticGrad(const mat p, const ucolvec z, const mat explanatoryVariables) {
    unsigned int nDeltas = explanatoryVariables.n_cols;

    colvec grad(2 * nDeltas, arma::fill::zeros);
    for (unsigned int j = 0; j < nDeltas; ++j) {
        double lowerGrad = 0;
        double upperGrad = 0;
        // NOTE(mike): using grad[j] += stops compiler from vectorising
        for (unsigned int i = 0; i < explanatoryVariables.n_rows; ++i) {
            lowerGrad += -explanatoryVariables.at(i, j) * p.at(i, 0) + (
                (z[i] == 2) ? explanatoryVariables.at(i, j) : 0
            );
            upperGrad += -explanatoryVariables.at(i, j) * p.at(i, 1) + (
                (z[i] == 3) ? explanatoryVariables.at(i, j) : 0
            );
        }
        grad[j] = lowerGrad;
        grad[nDeltas + j] = upperGrad;
    }

    return grad;
}

double logisticLogLikelihood(
    const mat delta, const ucolvec z, const mat explanatoryVariables
) {
    double logLikelihood = 0;

    for (unsigned int i = 0; i < z.n_elem; ++i) {
        double lowerSum = 0;
        double upperSum = 0;
        for (unsigned int j = 0; j < delta.n_cols; ++j) {
            lowerSum += delta.at(0, j) * explanatoryVariables.at(i, j);
            upperSum += delta.at(1, j) * explanatoryVariables.at(i, j);
        }

        if (z[i] == 2) {
            logLikelihood += lowerSum;
        } else if (z[i] == 3) {
            logLikelihood += upperSum;
        }

        // This is log(1 + exp(a) + exp(b))
        if (upperSum > lowerSum) {
            logLikelihood -= log1p(exp(lowerSum - upperSum) + exp(-upperSum)) + upperSum;
        } else {
            logLikelihood -= log1p(exp(upperSum - lowerSum) + exp(-lowerSum)) + lowerSum;
        }
    }

    return logLikelihood;
}

// [[Rcpp::export]]
void benchmarkLogistic(unsigned int nDeltas, unsigned int nValues, unsigned int nIterations) {
    mat deltaMatrix(2, nDeltas, arma::fill::randn);
    colvec deltaVector(2 * nDeltas, arma::fill::randn);
    mat explanatoryVariables(nValues, nDeltas, arma::fill::randn);
    ucolvec z = randi<ucolvec>(nValues, distr_param(1, 3));
    mat p = getLogisticP(deltaVector, explanatoryVariables);

    std::chrono::time_point<std::chrono::system_clock> startIteration, endIteration;
    std::chrono::duration<double> elapsed_seconds;

    Rcout << "getLogisticP matrix:\n";
    startIteration = std::chrono::system_clock::now();
    for (unsigned int iteration = 0; iteration < nIterations; ++iteration) {
        getLogisticP(deltaMatrix, explanatoryVariables);
    }
    endIteration = std::chrono::system_clock::now();
    elapsed_seconds = endIteration - startIteration;
    Rcout << "  Total time = " << (1000 * elapsed_seconds.count()) << "ms\n";
    Rcout << "  Time per iteration = " << (1000 * elapsed_seconds.count() / nIterations) << "ms\n";

    Rcout << "getLogisticP vector:\n";
    startIteration = std::chrono::system_clock::now();
    for (unsigned int iteration = 0; iteration < nIterations; ++iteration) {
        getLogisticP(deltaVector, explanatoryVariables);
    }
    endIteration = std::chrono::system_clock::now();
    elapsed_seconds = endIteration - startIteration;
    Rcout << "  Total time = " << (1000 * elapsed_seconds.count()) << "ms\n";
    Rcout << "  Time per iteration = " << (1000 * elapsed_seconds.count() / nIterations) << "ms\n";

    Rcout << "logisticHessian:\n";
    startIteration = std::chrono::system_clock::now();
    for (unsigned int iteration = 0; iteration < nIterations; ++iteration) {
        logisticHessian(p, explanatoryVariables);
    }
    endIteration = std::chrono::system_clock::now();
    elapsed_seconds = endIteration - startIteration;
    Rcout << "  Total time = " << (1000 * elapsed_seconds.count()) << "ms\n";
    Rcout << "  Time per iteration = " << (1000 * elapsed_seconds.count() / nIterations) << "ms\n";

    Rcout << "logisticGrad:\n";
    startIteration = std::chrono::system_clock::now();
    for (unsigned int iteration = 0; iteration < nIterations; ++iteration) {
        logisticGrad(p, z, explanatoryVariables);
    }
    endIteration = std::chrono::system_clock::now();
    elapsed_seconds = endIteration - startIteration;
    Rcout << "  Total time = " << (1000 * elapsed_seconds.count()) << "ms\n";
    Rcout << "  Time per iteration = " << (1000 * elapsed_seconds.count() / nIterations) << "ms\n";

    Rcout << "logisticLogLikelihood:\n";
    startIteration = std::chrono::system_clock::now();
    for (unsigned int iteration = 0; iteration < nIterations; ++iteration) {
        logisticLogLikelihood(deltaMatrix, z, explanatoryVariables);
    }
    endIteration = std::chrono::system_clock::now();
    elapsed_seconds = endIteration - startIteration;
    Rcout << "  Total time = " << (1000 * elapsed_seconds.count()) << "ms\n";
    Rcout << "  Time per iteration = " << (1000 * elapsed_seconds.count() / nIterations) << "ms\n";
}
