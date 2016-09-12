#include <map>
#include <set>
#include <vector>
#include <RcppArmadillo.h>
#include "rng.hpp"
#include "utils.hpp"

using arma::colvec;
using arma::mat;
using arma::max;
using arma::ucolvec;

using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

using ptsm::rng;

MissingValuesPair findMissingValues(Rcpp::NumericVector y) {
    int n = y.length();
    ucolvec yMissingIndices(n);
    std::vector<bool> yIsMissing(n);
    int nMissing = 0;
    for (int i = 0; i < n; ++i) {
        if (Rcpp::NumericVector::is_na(y[i])) {
            yMissingIndices[nMissing] = i;
            yIsMissing[i] = true;
            ++nMissing;
        } else {
            yIsMissing[i] = false;
        }
    }
    yMissingIndices.resize(nMissing);

    return std::make_pair(yMissingIndices, yIsMissing);
}

void sampleMissingY(
    colvec &y, colvec &logY, const ucolvec yMissingIndices, const ucolvec zCurrent,
    const std::vector<ParameterBoundDistribution> distributions
) {
    for (unsigned int i = 0; i < yMissingIndices.n_elem; ++i) {
        int index = yMissingIndices[i];
        if (zCurrent[index] == 1) {
            y[index] = 0;
        } else {
            y[index] = distributions[zCurrent[index] - 2].sample();
        }
        logY[index] = log(y[index]);
    }
}

unsigned int sampleSingleZ(const colvec p) {
    double u = rng.randu();

    for (unsigned int k = 0; k < p.n_elem; ++k) {
        if (u < p[k]) {
            return k + 2;
        }
        u -= p[k];
    }

    return 1;
}

ucolvec sampleZ(
    const mat pCurrent,
    const colvec y, const std::vector<bool> &yIsMissing,
    const std::vector<ParameterBoundDistribution> distributions
) {
    ucolvec z(y.n_elem);

    colvec p(pCurrent.n_cols);

    for (unsigned int i = 0; i < y.n_elem; ++i) {
        if (yIsMissing[i]) {
            z[i] = sampleSingleZ(pCurrent.row(i).t());
        } else {
            if (y[i] == 0) {
                z[i] = 1;
            } else {
                p.fill(0);
                for (unsigned int k = 0; k < p.n_elem; ++k) {
                    if (distributions[k].isInSupport(y[i])) {
                        p[k] = pCurrent(i, k) * distributions[k].pdf(y[i]);
                    }
                }

                double u = rng.randu() * arma::sum(p);
                for (unsigned int k = 0; k < p.n_elem; ++k) {
                    if (u < p[k]) {
                        z[i] = k + 2;
                        break;
                    }
                    u -= p[k];
                }
            }
        }
    }

    return z;
}

// [[Rcpp::export(name=".levels_to_list_integer_vector")]]
List levelsToListIntegerVector(IntegerVector x, IntegerVector xLevels) {
    std::set<int> levels(xLevels.begin(), xLevels.end());
    std::map<int, int> levelToListIndex;
    std::vector< std::vector<int> > output;

    for (int level : levels) {
        levelToListIndex[level] = output.size();
        output.push_back(std::vector<int>());
    }

    for (unsigned int i = 0; i < x.length(); ++i) {
        output[levelToListIndex[xLevels[i]]].push_back(x[i]);
    }

    return Rcpp::wrap(output);
}

// [[Rcpp::export(name=".levels_to_list_numeric_vector")]]
List levelsToListNumericVector(NumericVector x, IntegerVector xLevels) {
    std::set<int> levels(xLevels.begin(), xLevels.end());
    std::map<int, int> levelToListIndex;
    std::vector< std::vector<double> > output;

    for (int level : levels) {
        levelToListIndex[level] = output.size();
        output.push_back(std::vector<double>());
    }

    for (unsigned int i = 0; i < x.length(); ++i) {
        output[levelToListIndex[xLevels[i]]].push_back(x[i]);
    }

    return Rcpp::wrap(output);
}

// [[Rcpp::export(name=".levels_to_list_numeric_matrix")]]
List levelsToListNumericMatrix(NumericMatrix x, IntegerVector xLevels) {
    std::map<int, int> levels;
    std::map<int, int> levelToListIndex;
    List output;

    for (unsigned int i = 0; i < xLevels.length(); ++i) {
        if (levels.find(xLevels[i]) == levels.end()) {
            levels[xLevels[i]] = 0;
        }
        levels[xLevels[i]]++;
    }

    for (std::pair<int, int> level : levels) {
        levelToListIndex[level.first] = output.size();
        output.push_back(NumericMatrix(level.second, x.ncol()));

        // Set the level to 0 to be used as a row counter
        levels[level.first] = 0;
    }

    for (int i = 0; i < x.nrow(); ++i) {
        int level = xLevels[i];
        Rcpp::as<NumericMatrix>(output[levelToListIndex[level]]).row(levels[level]) = x.row(i);
        ++levels[level];
    }

    return output;
}
