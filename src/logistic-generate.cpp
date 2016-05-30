#include <RcppArmadillo.h>

#include "distribution.hpp"

using arma::colvec;
using arma::conv_to;
using arma::mat;
using arma::span;
using arma::ucolvec;

using Rcpp::as;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;
using Rcpp::StringVector;
using Rcpp::stop;
using Rcpp::wrap;

typedef struct {
    colvec y;
    ucolvec z;
} LogisticSample;

LogisticSample generate(
    colvec delta, mat explanatoryVariables,
    ParameterBoundDistribution lowerDistribution, ParameterBoundDistribution upperDistribution,
    unsigned int order
) {
    LogisticSample sample;
    sample.y = colvec(explanatoryVariables.n_rows);
    sample.z = ucolvec(explanatoryVariables.n_rows);

    unsigned int nExplanatoryVariables = explanatoryVariables.n_cols;
    unsigned int nDeltas = delta.n_elem / 2;

    ucolvec previousZs;
    if (order > 0) {
        previousZs = ucolvec(order);
        previousZs.fill(1);
    }

    colvec deltaLower = delta.head(nDeltas);
    colvec deltaUpper = delta.tail(nDeltas);

    for (unsigned int i = 0; i < explanatoryVariables.n_rows; ++i) {
        double lowerSum = dot(deltaLower.head(nExplanatoryVariables), explanatoryVariables.row(i));
        double upperSum = dot(deltaUpper.head(nExplanatoryVariables), explanatoryVariables.row(i));

        for (unsigned int j = 0; j < order; ++j) {
            if (previousZs[j] == 2) {
                lowerSum += deltaLower[nDeltas - 2 * j - 2];
                upperSum += deltaUpper[nDeltas - 2 * j - 2];
            } else if (previousZs[j] == 3) {
                lowerSum += deltaLower[nDeltas - 2 * j - 1];
                upperSum += deltaUpper[nDeltas - 2 * j - 1];
            }
        }

        double expDiff = exp(upperSum - lowerSum);
        double pLower = 1 / (1 + exp(-lowerSum) + expDiff);
        double pUpper = 1 / (1 + exp(-upperSum) + 1 / expDiff);

        double u = R::runif(0, 1);
        if (u < pLower) {
            sample.z[i] = 2;
            sample.y[i] = lowerDistribution.sample();
        } else if (u < pLower + pUpper) {
            sample.z[i] = 3;
            sample.y[i] = upperDistribution.sample();
        } else {
            sample.z[i] = 1;
            sample.y[i] = 0;
        }

        if (order > 0) {
            if (order > 1) {
                // Shift values back and add the new one
                previousZs(span(0, order - 2)) = previousZs(span(1, order - 1));
            }
            previousZs[order - 1] = sample.z[i];
        }
    }

    return sample;
}

// [[Rcpp::export(name=".ptsm_logistic_generate")]]
List logisticGenerate(
    NumericMatrix deltaR, NumericMatrix explanatoryVariablesR,
    NumericVector thetaLower, NumericVector thetaUpper, StringVector distributionNames,
    unsigned int order
) {
    Distribution lowerDistribution(distributionNames[0]);
    Distribution upperDistribution(distributionNames[1]);

    mat explanatoryVariables = as<mat>(explanatoryVariablesR);
    mat delta = as<mat>(deltaR);

    LogisticSample result = generate(
        conv_to<colvec>::from(vectorise(delta, 1).t()), explanatoryVariables,
        ParameterBoundDistribution(as<colvec>(thetaLower), lowerDistribution),
        ParameterBoundDistribution(as<colvec>(thetaUpper), upperDistribution),
        order
    );

    List output;
    output["y"] = wrap(result.y);
    output["z"] = wrap(result.z);
    return output;
}
