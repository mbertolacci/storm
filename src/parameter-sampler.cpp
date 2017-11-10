#include <RcppArmadillo.h>

#include "alpha-conjugate.hpp"
#include "parameter-sampler.hpp"
#include "rng.hpp"

using Rcpp::as;
using Rcpp::List;
using Rcpp::NumericVector;
using Rcpp::NumericMatrix;
using Rcpp::Rcout;
using Rcpp::stop;

using ptsm::rng;

ParameterSampler::ParameterSampler(List prior, List samplingScheme, Distribution distribution) {
    if (samplingScheme.containsElementNamed("type")) {
        useGibbs_ = strcmp(samplingScheme["type"], "gibbs") == 0;
    } else {
        useGibbs_ = distribution.getType() == GAMMA || distribution.getType() == LOG_NORMAL;
    }

    if (useGibbs_) {
        if (distribution.getType() == GAMMA) {
            if (prior.containsElementNamed("alpha")) {
                priorAlpha_ = as<arma::colvec>(prior["alpha"]);
            } else {
                stop("Must provide parameters for Gibbs sampler");
            }
            if (prior.containsElementNamed("beta")) {
                priorBeta_ = as<arma::colvec>(prior["beta"]);
            } else {
                stop("Must provide parameters for Gibbs sampler");
            }
        } else {
            if (prior.containsElementNamed("mu")) {
                priorMu_ = as<arma::colvec>(prior["mu"]);
            } else {
                stop("Must provide parameters for Gibbs sampler");
            }
            if (prior.containsElementNamed("tau")) {
                priorTau_ = as<arma::colvec>(prior["tau"]);
            } else {
                stop("Must provide parameters for Gibbs sampler");
            }
        }
    } else {
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
    }

    // Sanity checking
    if (useGibbs_ && !(distribution.getType() == GAMMA || distribution.getType() == LOG_NORMAL)) {
        stop("Gibbs sampler supported only for Gamma and Log-Normal distribution");
    }

    if (!useGibbs_) {
        if (!distribution.hasMaximumLikelihoodEstimate() && useMle_) {
            stop("No support for using MLE with %s", distribution.getName());
        }

        if (!distribution.hasHessian() && useObservedInformation_) {
            stop("No support for using Hessian with %s", distribution.getName());
        }
    }
}

arma::colvec ParameterSampler::sample(
    const arma::colvec currentParameters, const DataBoundDistribution& boundDistribution
) {
    if (useGibbs_) {
        return sampleGibbs_(currentParameters, boundDistribution);
    } else {
        return sampleMetropolisHastings_(currentParameters, boundDistribution);
    }
}

arma::colvec ParameterSampler::sampleGibbs_(
    const arma::colvec currentParameters, const DataBoundDistribution& boundDistribution
) {
    arma::colvec output(currentParameters);

    if (boundDistribution.getType() == GAMMA) {
        double n = static_cast<double>(boundDistribution.getN());

        output[0] = rGammaShapeConjugate(
            output[1],
            priorAlpha_[0] + boundDistribution.getSumLogY(),
            priorAlpha_[1] + n,
            priorAlpha_[2] + n
        );
        output[1] = 1 / rng.randg(
            priorBeta_[0] + output[0] * (priorAlpha_[1] + n),
            1 / (priorBeta_[1] + boundDistribution.getSumY())
        );
    } else if (boundDistribution.getType() == LOG_NORMAL) {
        double n = static_cast<double>(boundDistribution.getN());

        double muPrecision = priorMu_[1] + n * currentParameters[1];
        double muMean = (
            priorMu_[0] * priorMu_[1]
            + currentParameters[1] * boundDistribution.getSumLogY()
        ) / muPrecision;
        output[0] = muMean + rng.randn() / sqrt(muPrecision);

        double tauA = priorTau_[0] + n / 2;
        double tauB = (
            priorTau_[1]
            + 0.5 * (
                boundDistribution.getSumLogYSquared()
                - 2 * output[0] * boundDistribution.getSumLogY()
                + n * output[0] * output[0]
            )
        );
        output[1] = rng.randg(tauA, 1) / tauB;
    }

    return output;
}

double dlogmvtnorm(const arma::colvec x, const arma::colvec mu, const arma::mat cholesky) {
    return -arma::as_scalar(
        (x - mu).t() * cholesky.t() * cholesky * (x - mu)
    ) / 2.0;
}

arma::colvec ParameterSampler::sampleMetropolisHastings_(
    const arma::colvec currentParameters, const DataBoundDistribution& boundDistribution
) {
    arma::colvec proposalMean(currentParameters);

    if (useMle_) {
        proposalMean = boundDistribution.maximumLikelihoodEstimate(currentParameters);
    }

    if (useObservedInformation_) {
        arma::mat hessian = boundDistribution.hessian(proposalMean);
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

    arma::colvec proposalParameters = proposalMean + covarianceCholesky_ * rng.randn(currentParameters.n_elem);

    if (!satisfiesPrior_(proposalParameters)
        || boundDistribution.isInSupport(0, proposalParameters)) {
        // Reject the sample
        return currentParameters;
    }

    double currentLogDensity = boundDistribution.logLikelihood(currentParameters);
    double proposalLogDensity = boundDistribution.logLikelihood(proposalParameters);

    double qCurrentLogDensity = 0;
    double qProposalLogDensity = 0;
    if (useMle_) {
        // If using the MLE, the proposal distribution is not symmetric and therefore we must
        // calculate this
        qCurrentLogDensity = dlogmvtnorm(currentParameters, proposalMean, covarianceCholesky_);
        qProposalLogDensity = dlogmvtnorm(proposalParameters, proposalMean, covarianceCholesky_);
    }

    double alpha = proposalLogDensity - currentLogDensity + qCurrentLogDensity - qProposalLogDensity;
    if (log(rng.randu()) < alpha) {
        return proposalParameters;
    }

    return currentParameters;
}

bool ParameterSampler::satisfiesPrior_(arma::colvec parameters) const {
    for (unsigned int i = 0; i < parameters.n_elem; ++i) {
        if (parameters[i] < uniformPriorBounds_(i, 0)) return false;
        if (parameters[i] > uniformPriorBounds_(i, 1)) return false;
    }
    return true;
}
