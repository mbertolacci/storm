#ifndef SRC_LOGISTIC_PRIOR_HPP_
#define SRC_LOGISTIC_PRIOR_HPP_

#include <RcppArmadillo.h>

class LogisticParameterPrior {
 public:
    LogisticParameterPrior() = default;
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

    const arma::mat means() const {
        return means_;
    }

    const arma::mat variances() const {
        return variances_;
    }

 private:
    bool isUniform_;
    arma::mat uniformPriorBounds_;
    arma::mat means_;
    arma::mat variances_;
};

#endif  // SRC_LOGISTIC_PRIOR_HPP_
