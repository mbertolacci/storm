#include <RcppArmadillo.h>
#include "gengamma.hpp"
#include "rng.hpp"

using ptsm::rng;

double rgengamma(double mu, double sigma, double Q) {
    if (Q == 0) {
        return exp(mu + sigma * rng.randn());
    }
    return exp(
        mu + sigma * (
            log(Q * Q * rng.randg(1 / (Q * Q), 1)) / Q
        )
    );
}
