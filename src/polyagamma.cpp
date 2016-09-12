#include <RcppArmadillo.h>
#include "polyagamma.hpp"
#include "rng.hpp"

using Rcpp::NumericVector;
using ptsm::rng;

// TODO(mgnb): pick this for maximum performance. There's a trade-off between the accept/reject behaviour and the
// need to sample truncated inverse gauss random variates. I've put this lower already.
const double POLYAGAMMA_TRUNC = 0.2;

inline double square(double x) {
    return x * x;
}

double pnorm(double value) {
    return 0.5 * erfc(-value * M_SQRT1_2);
}

double pinvgauss(double x, double z, double lambda) {
    return (
        pnorm(sqrt(lambda / x) * (x * z - 1))
        + exp(2 * lambda * z) * pnorm(-sqrt(lambda / x) * (x * z + 1))
    );
}

double rinvtruncatedgauss(double z) {
    double X, U;

    if (z < 1 / POLYAGAMMA_TRUNC) {
        double alpha, E1, E2;

        do {
            do {
                E1 = rng.rande();
                E2 = rng.rande();
            } while (square(E1) > 2 * E2 / POLYAGAMMA_TRUNC);

            X = POLYAGAMMA_TRUNC / ((1 + POLYAGAMMA_TRUNC * E1) * (1 + POLYAGAMMA_TRUNC * E1));

            alpha = exp(-X * square(z) / 2);

            U = rng.randu();
        } while (U > alpha);
    } else {
        double mu = 1 / z;
        do {
            double Y = square(rng.randn());
            X = mu + 0.5 * square(mu) * Y - 0.5 * mu * sqrt(4 * mu * Y + square(mu * Y));

            U = rng.randu();
            if (U > mu / (mu + X)) {
                X = square(mu) / X;
            }
        } while (X > POLYAGAMMA_TRUNC);
    }

    return X;
}

double polyagammaCoeff(double x, unsigned int n) {
    double nPlusHalf = static_cast<double>(n) + 0.5;

    if (x <= POLYAGAMMA_TRUNC) {
        double twoOverPiX = 2 / (M_PI * x);
        return (
            M_PI * nPlusHalf * twoOverPiX * sqrt(twoOverPiX) * exp(
                -2 * square(nPlusHalf) / x
            )
        );
    } else {
        return (
            M_PI * nPlusHalf * exp(
                -square(nPlusHalf) * square(M_PI) * x / 2
            )
        );
    }
}

double rpolyagammaSingle(double z) {
    z = fabs(z) / 2;

    double K = square(M_PI) / 8 + square(z) / 2;
    double p = M_PI / (2 * K) * exp(-K * POLYAGAMMA_TRUNC);
    double q = 2 * exp(-z) * pinvgauss(POLYAGAMMA_TRUNC, z, 1.0);

    while (true) {
        double X;

        if (rng.randu() < p / (p + q)) {
            // Truncated exponential
            X = POLYAGAMMA_TRUNC + rng.rande() / K;
        } else {
            // Truncated inverse Gaussian
            X = rinvtruncatedgauss(z);
        }

        unsigned int n = 0;
        double S = polyagammaCoeff(X, 0);
        double Y = rng.randu() * S;

        while (true) {
            ++n;
            if (n % 2 == 1) {
                S -= polyagammaCoeff(X, n);
                if (Y <= S) {
                    return X / 4;
                }
            } else {
                S += polyagammaCoeff(X, n);
                if (Y > S) {
                    break;
                }
            }
        }
    }

    return 0;
}


double rpolyagamma(unsigned int n, double z) {
    double sum = 0;
    for (unsigned int i = 0; i < n; ++i) {
        sum += rpolyagammaSingle(z);
    }
    return sum;
}

// [[Rcpp::export(name="rpolyagamma")]]
NumericVector rpolyagammaVector(unsigned int length, unsigned int n, double z) {
    NumericVector output(length);
    for (unsigned int i = 0; i < length; ++i) {
        output[i] = rpolyagamma(n, z);
    }
    return output;
}
