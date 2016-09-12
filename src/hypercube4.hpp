#ifndef SRC_HYPERCUBE4_HPP_
#define SRC_HYPERCUBE4_HPP_

#include <algorithm>
#include <vector>
#include <RcppArmadillo.h>

class hypercube4 {
 public:
    hypercube4()
        : n_rows(0),
          n_cols(0),
          n_slices(0),
          n_hyperslices(0),
          n_elem(0) {}

    void set_size(unsigned int n_rows_, unsigned int n_cols_, unsigned int n_slices_, unsigned int n_hyperslices_) {
        n_rows = n_rows_;
        n_cols = n_cols_;
        n_slices = n_slices_;
        n_hyperslices = n_hyperslices_;
        n_elem = n_rows * n_cols * n_slices * n_hyperslices;

        values_.resize(n_elem);
    }

    void set_hyperslice(unsigned int index, const arma::cube input) {
        std::copy(
            input.begin(), input.end(),
            values_.begin() + (index * n_rows * n_cols * n_slices)
        );
    }

    Rcpp::NumericVector asNumericVector() {
        Rcpp::IntegerVector dimensions;
        dimensions.push_back(n_rows);
        dimensions.push_back(n_cols);
        dimensions.push_back(n_slices);
        dimensions.push_back(n_hyperslices);

        Rcpp::Dimension dim(dimensions);

        Rcpp::NumericVector output(dim);
        std::copy(values_.begin(), values_.end(), output.begin());

        return output;
    }

    unsigned int n_rows;
    unsigned int n_cols;
    unsigned int n_slices;
    unsigned int n_hyperslices;
    unsigned int n_elem;

 private:
    std::vector<double> values_;
};

#endif  // SRC_HYPERCUBE4_HPP_
