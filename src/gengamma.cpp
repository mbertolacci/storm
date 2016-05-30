#include <RcppArmadillo.h>
#include "rgamma-thread-safe.hpp"
#include "gengamma.hpp"

double rgengamma(double mu, double sigma, double Q) {
    if (Q == 0) {
        return R::rlnorm(mu, sigma);
    }
    return exp(
        mu + sigma * (
            log(Q * Q * rgammaThreadSafe(1 / (Q * Q), 1)) / Q
        )
    );
}
