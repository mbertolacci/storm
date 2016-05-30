#include <RcppArmadillo.h>
#include "logistic-utils.hpp"

using arma::colvec;
using arma::conv_to;
using arma::mat;
using arma::ucolvec;

using Rcpp::as;
using Rcpp::IntegerVector;
using Rcpp::NumericMatrix;
using Rcpp::stop;
using Rcpp::wrap;

// [[Rcpp::export(name=".ptsm_logistic_ergodic_p")]]
NumericMatrix logisticErgodicP(
    NumericMatrix deltaSamplesR, NumericMatrix zSamplesR, IntegerVector z0SamplesR,
    NumericMatrix explanatoryVariablesR, unsigned int order
) {
    mat deltaSamples = as<mat>(deltaSamplesR);
    mat zSamples = as<mat>(zSamplesR);
    ucolvec z0Samples = as<ucolvec>(z0SamplesR);

    if (deltaSamples.n_rows != zSamples.n_rows) {
        stop("Z and delta samples must be balanced");
    }
    if (order > 1) {
        stop("Only order 1 supported");
    }

    mat explanatoryVariables = as<mat>(explanatoryVariablesR);
    // Make room for the z's
    explanatoryVariables.resize(explanatoryVariables.n_rows, explanatoryVariables.n_cols + 2 * order);

    mat pHat(explanatoryVariables.n_rows, 2, arma::fill::zeros);
    unsigned int nDeltas = deltaSamples.n_cols / 2;

    #pragma omp parallel for
    for (unsigned int sample = 0; sample < deltaSamples.n_rows; ++sample) {
        explanatoryVariables(0, nDeltas - 2) = z0Samples[sample] == 2;
        explanatoryVariables(0, nDeltas - 1) = z0Samples[sample] == 3;
        for (unsigned int i = 0; i < zSamples.n_cols - 1; ++i) {
            explanatoryVariables(i + 1, nDeltas - 2) = zSamples(sample, i) == 2;
            explanatoryVariables(i + 1, nDeltas - 1) = zSamples(sample, i) == 3;
        }

        pHat += getLogisticP(
            conv_to<colvec>::from(deltaSamples.row(sample)),
            explanatoryVariables
        );
    }

    pHat /= static_cast<double>(deltaSamples.n_rows);

    mat pHatOut(explanatoryVariables.n_rows, 3);
    pHatOut.col(0) = 1 - pHat.col(0) - pHat.col(1);
    pHatOut.col(1) = pHat.col(0);
    pHatOut.col(2) = pHat.col(1);

    return wrap(pHatOut);
}

// [[Rcpp::export(name=".ptsm_logistic_predicted_p")]]
NumericMatrix logisticPredictedP(
    NumericMatrix deltaSamplesR, IntegerVector z0SamplesR, NumericMatrix explanatoryVariablesR, unsigned int order
) {
    mat deltaSamples = trans(as<mat>(deltaSamplesR));
    mat explanatoryVariables = as<mat>(explanatoryVariablesR);
    ucolvec z0Samples = as<ucolvec>(z0SamplesR);
    mat pHat(explanatoryVariables.n_rows, 3, arma::fill::zeros);
    unsigned int nDeltas = deltaSamples.n_rows / 2;

    for (unsigned int sampleIndex = 0; sampleIndex < deltaSamples.n_rows; ++sampleIndex) {
        colvec delta = deltaSamples.col(sampleIndex);
        colvec deltaLower = delta.head(nDeltas);
        colvec deltaUpper = delta.tail(nDeltas);

        double prevP1 = 0;
        double prevP2 = 0;
        double prevP3 = 0;
        if (z0Samples[sampleIndex] == 1) {
            prevP1 = 1;
        } else if (z0Samples[sampleIndex] == 2) {
            prevP2 = 1;
        } else if (z0Samples[sampleIndex] == 3) {
            prevP3 = 1;
        }

        for (unsigned int i = 0; i < explanatoryVariables.n_rows; ++i) {
            double lowerSum = dot(deltaLower.head(nDeltas - 2), explanatoryVariables.row(i));
            double upperSum = dot(deltaUpper.head(nDeltas - 2), explanatoryVariables.row(i));

            double exp21 = exp(lowerSum);
            double exp22 = exp(lowerSum + deltaLower[nDeltas - 2]);
            double exp23 = exp(lowerSum + deltaLower[nDeltas - 1]);
            double exp31 = exp(upperSum);
            double exp32 = exp(upperSum + deltaUpper[nDeltas - 2]);
            double exp33 = exp(upperSum + deltaUpper[nDeltas - 1]);

            double p11 = 1 / (1 + exp21 + exp31);
            double p12 = 1 / (1 + exp22 + exp32);
            double p13 = 1 / (1 + exp23 + exp33);

            double p21 = exp21 / (1 + exp21 + exp31);
            double p22 = exp22 / (1 + exp22 + exp32);
            double p23 = exp23 / (1 + exp23 + exp33);

            double p31 = exp31 / (1 + exp21 + exp31);
            double p32 = exp32 / (1 + exp22 + exp32);
            double p33 = exp33 / (1 + exp23 + exp33);

            double p1 = p11 * prevP1 + p12 * prevP2 + p13 * prevP3;
            double p2 = p21 * prevP1 + p22 * prevP2 + p23 * prevP3;
            double p3 = p31 * prevP1 + p32 * prevP2 + p33 * prevP3;

            pHat(i, 0) += p1;
            pHat(i, 1) += p2;
            pHat(i, 2) += p3;

            prevP1 = p1;
            prevP2 = p2;
            prevP3 = p3;
        }
    }

    pHat /= static_cast<double>(deltaSamples.n_rows);

    return wrap(pHat);
}
