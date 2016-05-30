#ifndef SRC_LOGISTIC_PARAMETER_SAMPLER_HPP_
#define SRC_LOGISTIC_PARAMETER_SAMPLER_HPP_

#include <RcppArmadillo.h>

class LogisticParameterPrior {
 public:
    explicit LogisticParameterPrior(Rcpp::List prior);

    bool isInSupport(const arma::mat delta) const;
    double logPdf(const arma::mat delta, const arma::mat means, const arma::mat variances) const;
    double logPdf(const arma::mat delta) const;

    arma::colvec grad(const arma::colvec delta, const arma::mat means, const arma::mat variances) const;
    arma::colvec grad(const arma::colvec delta) const;
    arma::colvec grad(const arma::mat delta, const arma::mat means, const arma::mat variances) const;
    arma::colvec grad(const arma::mat delta) const;

    arma::mat hessian(const arma::colvec delta, const arma::mat means, const arma::mat variances) const;
    arma::mat hessian(const arma::colvec delta) const;
    arma::mat hessian(const arma::mat delta, const arma::mat means, const arma::mat variances) const;
    arma::mat hessian(const arma::mat delta) const;

    LogisticParameterPrior withParameters(const arma::mat means, const arma::mat variances) {
        LogisticParameterPrior output(*this);
        output.means_ = means;
        output.variances_ = variances;
        return output;
    }

 private:
    bool isUniform_;
    arma::mat uniformPriorBounds_;
    arma::mat means_;
    arma::mat variances_;
};

class LogisticParameterSampler {
 public:
    explicit LogisticParameterSampler(Rcpp::List samplingScheme);

    void printAcceptanceRatios(int nIterations) const;
    void printAcceptanceRatios(int nIterations, int factor) const;
    void resetAcceptCounts();

    arma::mat sample(
        const arma::mat deltaCurrent, const arma::mat pCurrent,
        const arma::ucolvec zCurrent, const arma::mat explanatoryVariables,
        const LogisticParameterPrior prior
    );

 private:
    int accept_;
    double observedInformationInflationFactor_;
    bool ignoreCovariance_;
};

#endif  // SRC_LOGISTIC_PARAMETER_SAMPLER_HPP_
