    #ifndef SRC_RNG_HPP_
#define SRC_RNG_HPP_

#include <RcppArmadillo.h>

#include <cstdint>
#include <random>

namespace ptsm {

class RNG {
 public:
    RNG() {
        engine_ = std::mt19937_64(0);
    }

    explicit RNG(uint_fast64_t seed) {
        engine_ = std::mt19937_64(seed);
    }

    double randu() {
        return uniformDistribution_(engine_);
    }

    double randn() {
        return normalDistribution_(engine_);
    }

    arma::colvec randn(unsigned int n) {
        arma::colvec x(n);
        for (unsigned int i = 0; i < n; ++i) {
            x[i] = randn();
        }
        return x;
    }

    double rande() {
        return exponentialDistribution_(engine_);
    }

    double randg(double alpha, double beta) {
        std::gamma_distribution<double> distribution(alpha, beta);
        return distribution(engine_);
    }

    static void initialise();

 private:
    std::mt19937_64 engine_;
    std::uniform_real_distribution<double> uniformDistribution_;  // U(0, 1)
    std::normal_distribution<double> normalDistribution_;  // N(0, 1)
    std::exponential_distribution<double> exponentialDistribution_;  // exp(1)
};

#if defined(_OPENMP)
    extern thread_local RNG rng;
#else
    extern RNG rng;
#endif

}  // namespace ptsm

#endif  // SRC_RNG_HPP_
