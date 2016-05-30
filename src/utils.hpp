#ifndef SRC_UTILS_HPP_
#define SRC_UTILS_HPP_

#include <utility>
#include <vector>
#include <RcppArmadillo.h>
#include "distribution.hpp"

typedef std::pair< arma::ucolvec, std::vector<bool> > MissingValuesPair;
MissingValuesPair findMissingValues(Rcpp::NumericVector y);

void sampleMissingY(
    arma::colvec &y, arma::colvec &logY, const arma::ucolvec yMissingIndices, const arma::ucolvec zCurrent,
    const ParameterBoundDistribution boundLowerDistribution, const ParameterBoundDistribution boundUpperDistribution
);

int sampleSingleZ(double pLower, double pUpper);

arma::ucolvec sampleZ(
    const arma::mat pCurrent,
    const arma::colvec y, const std::vector<bool> yIsMissing,
    const ParameterBoundDistribution boundLowerDistribution, const ParameterBoundDistribution boundUpperDistribution
);

// Base case, once all arguments are consumed
template< typename OutputIt >
OutputIt copyMultiple(OutputIt destination) {
    return destination;
}

template< typename Input, typename OutputIt, typename... Args >
OutputIt copyMultiple(OutputIt destination, Input input, Args... args) {
    typename Input::iterator first = input.begin();
    typename Input::iterator last = input.end();

    while (first != last) {
        *destination = *first;
        ++destination;
        ++first;
    }

    return copyMultiple(
        destination,
        args...
    );
}

template <typename T>
inline arma::field<T> fieldFromList(Rcpp::List list) {
    arma::field<T> output(list.length());
    for (unsigned int i = 0; i < list.length(); ++i) {
        output[i] = Rcpp::as<T>(list[i]);
    }
    return output;
}

inline arma::cube cubeFromList(Rcpp::List list) {
    Rcpp::NumericMatrix base = list[0];
    arma::cube output(base.nrow(), base.ncol(), list.length());
    for (unsigned int i = 0; i < list.length(); ++i) {
        output.slice(i) = Rcpp::as<arma::mat>(list[i]);
    }
    return output;
}

inline arma::colvec vectorise(const arma::field<arma::colvec> input) {
    if (input.n_elem == 1) {
        return input[0];
    }

    unsigned int totalNElem = 0;
    for (unsigned int i = 0; i < input.n_elem; ++i) {
        totalNElem += input[i].n_elem;
    }

    arma::colvec output(totalNElem);
    double *outputPointer = output.memptr();
    for (unsigned int i = 0; i < input.n_elem; ++i) {
        std::memcpy(outputPointer, input[i].memptr(), sizeof(double) * input[i].n_elem);
        outputPointer += input[i].n_elem;
    }
    return output;
}

inline arma::ucolvec vectorise(const arma::field<arma::ucolvec> input) {
    if (input.n_elem == 1) {
        return input[0];
    }

    unsigned int totalNElem = 0;
    for (unsigned int i = 0; i < input.n_elem; ++i) {
        totalNElem += input[i].n_elem;
    }

    arma::ucolvec output(totalNElem);
    arma::uword *outputPointer = output.memptr();
    for (unsigned int i = 0; i < input.n_elem; ++i) {
        std::memcpy(outputPointer, input[i].memptr(), sizeof(arma::uword) * input[i].n_elem);
        outputPointer += input[i].n_elem;
    }
    return output;
}

inline arma::mat join_rows(const arma::field<arma::mat> input) {
    if (input.n_elem == 1) {
        return input[0];
    }

    arma::mat output = input[0];
    // NOTE(mike): potentially slow due to resizing output each time
    for (unsigned int i = 1; i < input.n_elem; ++i) {
        output = join_rows(output, input[i]);
    }
    return output;
}

inline arma::umat join_rows(const arma::field<arma::umat> input) {
    if (input.n_elem == 1) {
        return input[0];
    }

    arma::umat output = input[0];
    // NOTE(mike): potentially slow due to resizing output each time
    for (unsigned int i = 1; i < input.n_elem; ++i) {
        output = join_rows(output, input[i]);
    }
    return output;
}

inline arma::colvec vectorise(const arma::cube input, int dim1, int dim2, int dim3) {
    arma::colvec output(input.n_elem);

    if (dim1 == 2 && dim2 == 0 && dim3 == 1) {
        // slice -> column -> row
        unsigned int index = 0;
        for (unsigned int k = 0; k < input.n_slices; ++k) {
            for (unsigned int i = 0; i < input.n_rows; ++i) {
                for (unsigned int j = 0; j < input.n_cols; ++j) {
                    output[index++] = input.at(i, j, k);
                }
            }
        }
    }

    return output;
}

template <typename T>
inline Rcpp::List listFromField(const arma::field<T> input) {
    Rcpp::List output;
    for (unsigned int i = 0; i < input.n_elem; ++i) {
        output.push_back(Rcpp::wrap(input[i]));
    }
    return output;
}

inline arma::mat sum(const arma::field<arma::mat> input) {
    if (input.n_elem == 1) {
        return input[0];
    }
    arma::mat output(arma::size(input[0]), arma::fill::zeros);
    for (unsigned int i = 0; i < input.n_elem; ++i) {
        output += input[i];
    }
    return output;
}

#endif  // SRC_UTILS_HPP_
