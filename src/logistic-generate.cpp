#include <RcppArmadillo.h>

#include "distribution.hpp"
#include "rng.hpp"

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

using ptsm::RNG;

typedef struct {
    colvec y;
    ucolvec z;
} LogisticSample;

LogisticSample generate(
    const mat delta, const mat explanatoryVariables,
    const std::vector<ParameterBoundDistribution> distributions, unsigned int order
) {
    LogisticSample sample;
    sample.y = colvec(explanatoryVariables.n_rows);
    sample.z = ucolvec(explanatoryVariables.n_rows);

    unsigned int nExplanatoryVariables = explanatoryVariables.n_cols;
    unsigned int nDeltas = delta.n_cols;
    unsigned int nComponents = distributions.size();

    ucolvec previousZs;
    if (order > 0) {
        previousZs = ucolvec(order);
        previousZs.fill(1);
    }

    colvec componentSums(nComponents + 1, arma::fill::zeros);

    for (unsigned int i = 0; i < explanatoryVariables.n_rows; ++i) {
        for (unsigned int k = 0; k < nComponents; ++k) {
            componentSums[k] = dot(
                delta.row(k).head(nExplanatoryVariables),
                explanatoryVariables.row(i)
            );

            for (unsigned int j = 0; j < order; ++j) {
                if (previousZs[j] == 1) continue;

                unsigned int previousComponent = previousZs[j] - 2;
                componentSums[k] += delta(
                    k,
                    nDeltas - nComponents * j + previousComponent - nComponents
                );
            }
        }

        colvec componentExpDiff = exp(componentSums - arma::max(componentSums));
        double denominator = arma::sum(componentExpDiff);

        double u = rng.randu();

        sample.z[i] = 1;
        sample.y[i] = 0;
        for (unsigned int k = 0; k < nComponents; ++k) {
            double p = componentExpDiff[k] / denominator;
            // NOTE(mgnb): if none of these comparisons match, we leave z == 1
            if (u < p) {
                sample.z[i] = k + 2;
                sample.y[i] = distributions[k].sample();
                break;
            }
            u -= p;
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
    List distributionParameters, StringVector distributionNames,
    unsigned int order
) {
    RNG::initialise();

    std::vector<ParameterBoundDistribution> distributions;
    for (unsigned int k = 0; k < distributionNames.length(); ++k) {
        distributions.push_back(ParameterBoundDistribution(
            as<colvec>(distributionParameters[k]),
            Distribution(distributionNames[k])
        ));
    }

    mat explanatoryVariables = as<mat>(explanatoryVariablesR);
    mat delta = as<mat>(deltaR);

    LogisticSample result = generate(
        delta, explanatoryVariables,
        distributions, order
    );

    List output;
    output["y"] = wrap(result.y);
    output["z"] = wrap(result.z);
    return output;
}
