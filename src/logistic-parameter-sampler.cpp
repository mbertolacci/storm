#include <algorithm>
#include <RcppArmadillo.h>

#include "logistic-mle.hpp"
#include "logistic-parameter-sampler.hpp"
#include "logistic-prior.hpp"
#include "logistic-utils.hpp"
#include "polyagamma.hpp"
#include "rng.hpp"

using arma::colvec;
using arma::mat;
using arma::max;
using arma::ucolvec;

using Rcpp::as;
using Rcpp::List;
using Rcpp::Rcout;

using ptsm::rng;

LogisticParameterSampler::LogisticParameterSampler(List samplingScheme)
    : accept_(0) {
    if (samplingScheme.containsElementNamed("method")) {
        if (strcmp(samplingScheme["method"], "metropolis_hastings") == 0) {
            isPolson_ = false;
        } else {
            isPolson_ = true;
        }
    } else {
        isPolson_ = true;
    }

    if (samplingScheme.containsElementNamed("observed_information_inflation_factor")) {
        observedInformationInflationFactor_ = samplingScheme["observed_information_inflation_factor"];
    } else {
        observedInformationInflationFactor_ = 1.0;
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

colvec sampleDeltaComponent(
    const mat& deltaCurrent, const ucolvec& zCurrent, const mat& explanatoryVariables,
    const mat& sums, const mat& expSums, const mat& maxSums,
    const colvec& priorMean, const colvec& priorVariance,
    unsigned int component
) {
    unsigned int nDeltas = explanatoryVariables.n_cols;
    colvec omega(explanatoryVariables.n_rows);
    colvec meanPart(explanatoryVariables.n_rows);

    for (unsigned int i = 0; i < explanatoryVariables.n_rows; ++i) {
        double thisSum = sums(component, i);
        // This is log(1 + \sum_{k' != component} exp(sum[k']))
        double c = log(arma::sum(expSums.col(i)) - expSums(component, i)) + maxSums[i];

        omega[i] = rpolyagamma(1, thisSum - c);
        meanPart[i] = (zCurrent[i] == (component + 2) ? 1.0 : 0) - 0.5 + omega[i] * c;
    }

    mat invPriorVariance = diagmat(1 / priorVariance);
    // R'R = X' \Omega X + V^{-1}
    mat R = chol(explanatoryVariables.t() * (explanatoryVariables.each_col() % omega) + invPriorVariance);
    // R'z = X' m_n + V^{-1} m_0
    colvec z = solve(trimatl(R.t()), explanatoryVariables.t() * meanPart + invPriorVariance * priorMean);
    // R m = z
    colvec mean = solve(trimatu(R), z);

    return mean + solve(trimatu(R), rng.randn(nDeltas));
}

mat LogisticParameterSampler::samplePolson(
    const mat& deltaCurrent, const ucolvec& zCurrent, const mat& explanatoryVariables,
    const LogisticParameterPrior& prior
) {
    mat deltaNew(deltaCurrent);

    colvec maxSums(explanatoryVariables.n_rows);
    // The last column is for the 0 given by component set to 0
    mat sums(deltaCurrent.n_rows + 1, explanatoryVariables.n_rows, arma::fill::zeros);
    mat expSums(deltaCurrent.n_rows + 1, explanatoryVariables.n_rows, arma::fill::zeros);
    for (unsigned int i = 0; i < explanatoryVariables.n_rows; ++i) {
        for (unsigned int k = 0; k < deltaCurrent.n_rows; ++k) {
            sums(k, i) = arma::dot(deltaCurrent.row(k), explanatoryVariables.row(i));
        }

        maxSums[i] = arma::max(sums.col(i));
        expSums.col(i) = exp(sums.col(i) - maxSums[i]);
    }

    for (unsigned int k = 0; k < deltaCurrent.n_rows; ++k) {
        deltaNew.row(k) = sampleDeltaComponent(
            deltaNew, zCurrent, explanatoryVariables,
            sums, expSums, maxSums,
            prior.means().row(k).t(), prior.variances().row(k).t(),
            k
        ).t();

        for (unsigned int i = 0; i < explanatoryVariables.n_rows; ++i) {
            sums(k, i) = arma::dot(deltaNew.row(k), explanatoryVariables.row(i));
            expSums(k, i) = exp(sums(k, i) - maxSums[i]);
        }
    }

    return deltaNew;
}

mat LogisticParameterSampler::sampleMetropolisHastings(
    const mat& deltaCurrent, const ucolvec& zCurrent, const mat& explanatoryVariables,
    const LogisticParameterPrior& prior
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
            unitNormals[i] = rng.randn();
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

    // TODO(mgnb): put the proposal distribution here!

    double alpha = proposalLogDensity - currentLogDensity;
    if (log(rng.randu()) < alpha) {
        accept_++;
        return deltaProposal;
    }

    return deltaCurrent;
}

mat LogisticParameterSampler::sample(
    const mat& deltaCurrent, const ucolvec& zCurrent, const mat& explanatoryVariables,
    const LogisticParameterPrior& prior
) {
    if (isPolson_) {
        return samplePolson(deltaCurrent, zCurrent, explanatoryVariables, prior);
    } else {
        return sampleMetropolisHastings(deltaCurrent, zCurrent, explanatoryVariables, prior);
    }
}
