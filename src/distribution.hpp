#ifndef SRC_DISTRIBUTION_HPP_
#define SRC_DISTRIBUTION_HPP_

#include <RcppArmadillo.h>
#include "gengamma.hpp"
#include "gev.hpp"
#include "rng.hpp"

using ptsm::rng;

typedef enum {
    GAMMA,
    GEV,
    GENERALISED_GAMMA,
    LOG_NORMAL
} DistributionType;

class Distribution {
 public:
    Distribution() : type_(GAMMA) {}

    explicit Distribution(const char *typeName) {
        if (strcmp(typeName, "gamma") == 0) {
            type_ = GAMMA;
        } else if (strcmp(typeName, "gev") == 0) {
            type_ = GEV;
        } else if (strcmp(typeName, "gengamma") == 0) {
            type_ = GENERALISED_GAMMA;
        } else if (strcmp(typeName, "lnorm") == 0) {
            type_ = LOG_NORMAL;
        } else {
            Rcpp::stop("Distribution %s not supported", typeName);
        }
    }

    const char *getName() const {
        switch (type_) {
        case GAMMA:
            return "gamma";
        case GEV:
            return "gev";
        case GENERALISED_GAMMA:
            return "gengamma";
        case LOG_NORMAL:
            return "lnorm";
        default:
            return NULL;
        }
    }

    DistributionType getType() const { return type_; }

    bool hasMaximumLikelihoodEstimate() const {
        return type_ != GEV;
    }

    bool hasHessian() const {
        return true;
    }

    bool isInSupport(double x, arma::colvec parameters) const {
        switch (type_) {
        case GAMMA:
            return x > 0;
        case GEV:
            if (parameters[2] == 0) {
                return true;
            } else if (parameters[2] < 0) {
                return x < parameters[0] - parameters[1] / parameters[2];
            } else {
                return x > parameters[0] - parameters[1] / parameters[2];
            }
        case GENERALISED_GAMMA:
            return x > 0;
        case LOG_NORMAL:
            return x > 0;
        default:
            return false;
        }
    }

    double sample(arma::colvec parameters) const {
        switch (type_) {
        case GAMMA:
            return rng.randg(parameters[0], parameters[1]);
        case GEV:
            return rgev(parameters[0], parameters[1], parameters[2]);
        case GENERALISED_GAMMA:
            return rgengamma(parameters[0], parameters[1], parameters[2]);
        case LOG_NORMAL:
            return exp(parameters[0] + rng.randn() / sqrt(parameters[1]));
        default:
            return -1;
        }
    }

 private:
    DistributionType type_;
};

class DataBoundDistribution {
 public:
    DataBoundDistribution(
        arma::colvec y, arma::colvec logY, arma::ucolvec z, unsigned int thisZ, Distribution distribution
    );

    DataBoundDistribution(
        unsigned int thisN, double sumY, double sumLogY, double sumLogYSquared, Distribution distribution
    );

    double logLikelihood(arma::colvec parameters) const;

    bool hasMaximumLikelihoodEstimate() const {
        return distribution_.hasMaximumLikelihoodEstimate();
    }

    bool hasHessian() const {
        return distribution_.hasHessian();
    }

    bool isInSupport(double x, arma::colvec parameters) const {
        return distribution_.isInSupport(x, parameters);
    }

    DistributionType getType() const { return distribution_.getType(); }

    arma::colvec maximumLikelihoodEstimate(arma::colvec start) const;

    arma::mat hessian(arma::colvec parameters) const;

    double getSumY() const {
        return sumY_;
    }

    double getSumLogY() const {
        return sumLogY_;
    }

    double getSumLogYSquared() const {
        return sumLogYSquared_;
    }

    unsigned int getN() const {
        return thisN_;
    }

 private:
    arma::colvec y_;
    arma::colvec logY_;
    arma::ucolvec z_;
    unsigned int thisZ_;
    unsigned int thisN_;
    double sumY_;
    double sumLogY_;
    double sumLogYSquared_;
    Distribution distribution_;

    arma::colvec2 genGammaEllPrime(arma::colvec2 parameters) const;
};

class ParameterBoundDistribution {
 public:
    ParameterBoundDistribution(arma::colvec parameters, Distribution distribution)
        : parameters_(parameters),
          distribution_(distribution) {
        if (distribution_.getType() == GAMMA) {
            norm_ = pow(parameters_[1], -parameters_[0]) / tgamma(parameters_[0]);
        } else if (distribution_.getType() == LOG_NORMAL) {
            norm_ = -0.5 * log(parameters_[1]) + 0.5 * log(2 * M_PI);
        } else if (distribution_.getType() == GENERALISED_GAMMA) {
            double sigma = parameters_[1];
            double Q = parameters_[2];
            double invQSquared = 1 / (Q * Q);
            norm_ = log(fabs(Q)) + invQSquared * log(invQSquared) - log(sigma) - lgamma(invQSquared);
        }
    }

    bool isInSupport(double x) const {
        return distribution_.isInSupport(x, parameters_);
    }

    double pdf(double x) const {
        switch (distribution_.getType()) {
        case GAMMA:
            return norm_ * pow(x, parameters_[0] - 1) * exp(-x / parameters_[1]);
        case GEV:
            return dgev(x, parameters_[0], parameters_[1], parameters_[2], false);
        case GENERALISED_GAMMA: {
            if (parameters_[2] == 0) {
                return R::dlnorm(x, parameters_[0], parameters_[1], false);
            }
            double Q = parameters_[2];
            double Qw = Q * (log(x) - parameters_[0]) / parameters_[1];
            return exp(norm_ + (Qw - exp(Qw)) / (Q * Q)) / x;
        }
        case LOG_NORMAL: {
            double mu = parameters_[0];
            double tau = parameters_[1];
            double logX = log(x);
            return exp(-logX - tau * (logX - mu) * (logX - mu) / 2 - norm_);
        } default:
            return -1;
        }
    }

    double sample() const {
        return distribution_.sample(parameters_);
    }

 private:
    arma::colvec parameters_;
    Distribution distribution_;

    double norm_;
};

#endif  // SRC_DISTRIBUTION_HPP_
