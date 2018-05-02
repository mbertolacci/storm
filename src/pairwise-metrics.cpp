#include <RcppArmadillo.h>

using arma::colvec;
using arma::datum;
using arma::field;
using arma::imat;
using arma::mat;
using arma::ucolvec;
using arma::umat;
using Rcpp::LogicalMatrix;
using Rcpp::NumericMatrix;

colvec averageRanks(const colvec& input, const ucolvec& sortIndices) {
    colvec output(input.n_elem);

    unsigned int i = 0;
    while (i < sortIndices.n_elem) {
        unsigned int runEnd = i + 1;
        // Find the end of the current run (or the end of the array)
        while (runEnd < sortIndices.n_elem && input[sortIndices[i]] == input[sortIndices[runEnd]]) {
            ++runEnd;
        }
        // Set all values to the average rank
        double averageRank = static_cast<double>(i + 1 + runEnd) / 2.0;
        for (; i < runEnd; ++i) {
            output[sortIndices[i]] = averageRank;
        }
    }
    return output;
}

colvec averageRanks(const ucolvec& sortIndices, const ucolvec& selectedIndices, const colvec& input) {
    ucolvec indexMap(input.n_elem);
    unsigned int empty = input.n_elem + 1;
    indexMap.fill(empty);
    for (unsigned int i = 0; i < selectedIndices.n_elem; ++i) {
        indexMap[selectedIndices[i]] = i;
    }
    ucolvec selectedSortIndices(selectedIndices.n_elem);
    unsigned int j = 0;
    for (unsigned int i = 0; i < sortIndices.n_elem; ++i) {
        if (indexMap[sortIndices[i]] == empty) continue;
        selectedSortIndices[j] = indexMap[sortIndices[i]];
        ++j;
    }

    return averageRanks(input.elem(selectedIndices), selectedSortIndices);
}

//' @export
// [[Rcpp::export(name="spearman_pairwise_correlation")]]
NumericMatrix spearmanPairwiseCorrelation(NumericMatrix inputR) {
    mat input = Rcpp::as<mat>(inputR);
    unsigned int n = input.n_cols;
    mat output = mat(n, n, arma::fill::ones);

    field<ucolvec> finiteIndices(n);
    for (unsigned int i = 0; i < n; ++i) {
        finiteIndices[i] = arma::find_finite(input.col(i));
    }

    // HACK(mgnb): replace NaN with Inf to make sort_index work
    input.replace(datum::nan, datum::inf);
    field<ucolvec> sortIndices(n);
    for (unsigned int i = 0; i < n; ++i) {
        sortIndices[i] = arma::sort_index(input.col(i));
    }

    #pragma omp parallel for if (n > 10) schedule(dynamic, 1)
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = i + 1; j < n; ++j) {
            // Find the intersection between the indices
            std::vector<unsigned int> finitePairsVector;
            std::set_intersection(
                finiteIndices[i].begin(), finiteIndices[i].end(),
                finiteIndices[j].begin(), finiteIndices[j].end(),
                std::back_inserter(finitePairsVector)
            );
            ucolvec finitePairs(finitePairsVector);

            if (finitePairs.n_elem <= 1) {
                output(i, j) = NA_REAL;
                output(j, i) = NA_REAL;
                continue;
            }

            colvec rankI = averageRanks(sortIndices[i], finitePairs, input.col(i));
            colvec rankJ = averageRanks(sortIndices[j], finitePairs, input.col(j));

            output(i, j) = arma::as_scalar(arma::cor(rankI, rankJ));
            output(j, i) = output(i, j);
        }
    }
    return Rcpp::wrap(output);
}

// Given two sorted vectors, all unique, find the number of items common
// between them
unsigned int setIntersectionCount(const ucolvec& left, const ucolvec& right) {
    unsigned int count = 0, i = 0, j = 0;
    while (i < left.size() && j < right.size()) {
        if (left[i] < right[j]) {
            ++i;
        } else if (left[i] > right[j]) {
            ++j;
        } else {
            ++i;
            ++j;
            ++count;
        }
    }

    return count;
}

//' @export
// [[Rcpp::export(name="pairwise_conditional_probabilities")]]
NumericMatrix pairwiseConditionalProbabilities(LogicalMatrix inputR) {
    imat input = Rcpp::as<imat>(inputR);
    unsigned int n = input.n_cols;

    field<ucolvec> finiteIndices(n);
    for (unsigned int i = 0; i < n; ++i) {
        finiteIndices[i] = arma::find(input.col(i) != NA_LOGICAL);
    }

    field<ucolvec> trueIndices(n);
    for (unsigned int i = 0; i < n; ++i) {
        trueIndices[i] = arma::find(input.col(i) == 1);
    }

    mat output = mat(n, n, arma::fill::ones);

    #pragma omp parallel for if (n > 10) schedule(dynamic, 1)
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = i + 1; j < n; ++j) {
            unsigned int truePairsCount = setIntersectionCount(
                trueIndices[i],
                trueIndices[j]
            );
            unsigned int rightTrueCount = setIntersectionCount(
                finiteIndices[i],
                trueIndices[j]
            );

            output(i, j) = (
                static_cast<double>(truePairsCount)
                / static_cast<double>(rightTrueCount)
            );
            output(j, i) = output(i, j);
        }
    }


    return Rcpp::wrap(output);
}

//' @export
// [[Rcpp::export(name="pairwise_log_odds_of_match")]]
NumericMatrix pairwiseLogOddsOfMatch(LogicalMatrix inputR) {
    imat input = Rcpp::as<imat>(inputR);
    unsigned int n = input.n_cols;

    field<ucolvec> finiteIndices(n);
    for (unsigned int i = 0; i < n; ++i) {
        finiteIndices[i] = arma::find(input.col(i) != NA_LOGICAL);
    }

    field<ucolvec> falseIndices(n);
    for (unsigned int i = 0; i < n; ++i) {
        falseIndices[i] = arma::find(input.col(i) == 0);
    }

    field<ucolvec> trueIndices(n);
    for (unsigned int i = 0; i < n; ++i) {
        trueIndices[i] = arma::find(input.col(i) == 1);
    }

    mat output = mat(n, n, arma::fill::ones);

    #pragma omp parallel for if (n > 10) schedule(dynamic, 1)
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = i + 1; j < n; ++j) {
            unsigned int falsePairsCount = setIntersectionCount(
                falseIndices[i],
                falseIndices[j]
            );
            unsigned int truePairsCount = setIntersectionCount(
                trueIndices[i],
                trueIndices[j]
            );
            unsigned int finitePairsCount = setIntersectionCount(
                finiteIndices[i],
                finiteIndices[j]
            );

            output(i, j) = (
                std::log(static_cast<double>(falsePairsCount + truePairsCount))
                - std::log(static_cast<double>(finitePairsCount - falsePairsCount - truePairsCount))
            );
            output(j, i) = output(i, j);
        }
    }

    return Rcpp::wrap(output);
}
