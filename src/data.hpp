#ifndef SRC_DATA_HPP_
#define SRC_DATA_HPP_

#include <RcppArmadillo.h>
#include <vector>

namespace ptsm {

class Data {
 public:
    const unsigned int n_elem;

    Data()
        : n_elem(0) {}

    Data(Rcpp::NumericVector values)
        : n_elem(values.length()) {
        values_ = Rcpp::as<arma::colvec>(values);
        logValues_ = arma::log(values_);

        missingIndices_.set_size(values_.n_elem);
        isMissing_.resize(values_.n_elem);

        unsigned int nMissing = 0;
        for (unsigned int i = 0; i < values.length(); ++i) {
            if (Rcpp::NumericVector::is_na(values[i])) {
                missingIndices_[nMissing] = i;
                isMissing_[i] = true;
                ++nMissing;
            } else {
                isMissing_[i] = false;
            }
        }

        missingIndices_.set_size(nMissing);
    }

    double& operator[](unsigned int i) {
        return values_[i];
    }

    arma::colvec operator()(const arma::ucolvec indices) {
        return values_(indices);
    }

    double log(unsigned int i) const {
        return logValues_[i];
    }

    const arma::ucolvec& missingIndices() const {
        return missingIndices_;
    }

    bool isMissing(unsigned int i) const {
        return isMissing_[i];
    }

 private:
    arma::colvec values_;
    arma::colvec logValues_;
    arma::ucolvec missingIndices_;
    std::vector<bool> isMissing_;
};

}  // namespace ptsm

#endif  // SRC_DATA_HPP_
