#include <RcppArmadillo.h>

#include "parameter-sampler.hpp"
#include "rng.hpp"

using Rcpp::as;
using Rcpp::List;
using Rcpp::NumericVector;
using Rcpp::NumericMatrix;
using Rcpp::Rcout;
using Rcpp::stop;

using ptsm::rng;

bool acceptOrReject(double currentLogDensity, double proposalLogDensity) {
    double alpha = proposalLogDensity - currentLogDensity;
    return log(rng.randu()) < alpha;
}

ParameterSampler::ParameterSampler(List prior, List samplingScheme, Distribution distribution)
    : accept_{0} {

    if (samplingScheme.containsElementNamed("use_mle")) {
        useMle_ = static_cast<int>(samplingScheme["use_mle"]);
    } else {
        useMle_ = distribution.hasMaximumLikelihoodEstimate();
    }

    if (samplingScheme.containsElementNamed("use_observed_information")) {
        useObservedInformation_ = static_cast<int>(samplingScheme["use_observed_information"]);
    } else {
        useObservedInformation_ = distribution.hasHessian();
    }

    if (samplingScheme.containsElementNamed("ignore.covariance")) {
        ignoreCovariance_ = static_cast<int>(samplingScheme["ignore.covariance"]);
    } else {
        ignoreCovariance_ = false;
    }

    if (samplingScheme.containsElementNamed("observed_information_inflation_factor")) {
        observedInformationInflationFactor_ = samplingScheme["observed_information_inflation_factor"];
    } else {
        observedInformationInflationFactor_ = 1.0;
    }

    if (strcmp(prior["type"], "uniform") == 0) {
        uniformPriorBounds_ = as<arma::mat>(prior["bounds"]);
    }

    if (!useObservedInformation_) {
        covarianceCholesky_ = arma::chol(as<arma::mat>(samplingScheme["covariance"]));
    }

    // Sanity checking
    if (!distribution.hasMaximumLikelihoodEstimate() && useMle_) {
        stop("No support for using MLE with %s", distribution.getName());
    }

    if (!distribution.hasHessian() && useObservedInformation_) {
        stop("No support for using Hessian with %s", distribution.getName());
    }
}

void ParameterSampler::printAcceptanceRatio(int nIterations) {
    Rcout << "Acceptance ratios " << (static_cast<double>(accept_) / static_cast<double>(nIterations)) << "\n";
}

void ParameterSampler::resetAcceptCount() {
    accept_ = 0;
}

arma::colvec ParameterSampler::sample(
    const arma::colvec currentParameters, const DataBoundDistribution &boundDistribution
) {
    arma::colvec proposalMean(currentParameters);

    if (useMle_) {
        proposalMean = boundDistribution.maximumLikelihoodEstimate(currentParameters);
    }

    arma::colvec proposalParameters(currentParameters.n_elem);
    arma::colvec unitNormals(currentParameters.n_elem);

    if (useObservedInformation_) {
        arma::mat hessian = boundDistribution.hessian(proposalMean);
        if (ignoreCovariance_) {
            hessian = arma::diagmat(hessian);
        }

        arma::mat negInverseHessian;

        try {
            negInverseHessian = (-hessian).i();
        } catch (std::runtime_error &error) {
            Rcout << proposalMean << "\n";
            Rcout << hessian << "\n";
            throw;
        }

        try {
            covarianceCholesky_ = arma::chol(observedInformationInflationFactor_ * negInverseHessian);
        } catch (std::runtime_error &error) {
            Rcout << proposalMean << "\n";
            Rcout << hessian << "\n";
            throw;
        }
    }

    while (true) {
        for (unsigned int i = 0; i < currentParameters.n_elem; ++i) {
            unitNormals[i] = rng.randn();
        }
        proposalParameters = proposalMean + covarianceCholesky_ * unitNormals;

        if (satisfiesPrior(proposalParameters)
            && !boundDistribution.isInSupport(0, proposalParameters)) {
            break;
        }
    }

    double currentLogDensity = boundDistribution.logLikelihood(currentParameters);
    double proposalLogDensity = boundDistribution.logLikelihood(proposalParameters);

    double alpha = proposalLogDensity - currentLogDensity;
    if (log(rng.randu()) < alpha) {
        accept_++;
        return proposalParameters;
    }

    return currentParameters;
}

bool ParameterSampler::satisfiesPrior(arma::colvec parameters) const {
    for (unsigned int i = 0; i < parameters.n_elem; ++i) {
        if (parameters[i] < uniformPriorBounds_(i, 0)) return false;
        if (parameters[i] > uniformPriorBounds_(i, 1)) return false;
    }
    return true;
}
