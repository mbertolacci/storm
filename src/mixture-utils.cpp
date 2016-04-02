#include <Rcpp.h>
#include "mixture-utils.h"

using namespace Rcpp;

// Return an integer in [0, weights.length() - 1], weighted by weights
int randomWeightedIndex(NumericVector weights) {
    int k = weights.length();
    double u = R::runif(0, sum(weights));
    for (int j = 0; j < k; ++j) {
        u -= weights[j];
        if (u < 0) {
            return j;
        }
    }
    // Very unlikely to reach here, would need u == sum(weights), but just in case.
    return k - 1;
}
