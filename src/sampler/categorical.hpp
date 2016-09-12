#ifndef SRC_SAMPLER_CATEGORICAL_HPP_
#define SRC_SAMPLER_CATEGORICAL_HPP_

#include <RcppArmadillo.h>
#include <random>

namespace ptsm {

class CategoricalSampler {
public:
    CategoricalSampler(const arma::colvec weights)
        : weights_(weights) {}

    template<typename _RandomNumberEngine>
    unsigned int operator()(_RandomNumberEngine engine) const {
        // NOTE(mgnb): if some weight is infinite, return it straight away - allows the use of Dirac
        // delta functions
        for (unsigned int k = 0; k < weights_.n_elem; ++k) {
            if (weights_[k] == std::numeric_limits<double>::infinity()) {
                return k;
            }
        }

        double sum = arma::sum(weights_);
        double u = std::uniform_real_distribution<double>(0, sum)(engine);

        for (unsigned int k = 0; k < weights_.n_elem; ++k) {
            if (u < weights_[k]) {
                return k;
            }
            u -= weights_[k];
        }

        // Maybe we have an underflow; in any case, return the last one
        return weights_.n_elem - 1;
    }

private:
    arma::colvec weights_;
};

}  // namespace ptsm

#endif  // SRC_SAMPLER_CATEGORICAL_HPP_
