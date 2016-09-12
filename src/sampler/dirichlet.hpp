#ifndef SRC_SAMPLER_DIRICHLET_HPP_
#define SRC_SAMPLER_DIRICHLET_HPP_

#include <RcppArmadillo.h>
#include <random>

namespace ptsm {

class DirichletSampler {
public:
    DirichletSampler(const arma::colvec parameters)
        : parameters_(parameters) {}

    template<typename _RandomNumberEngine>
    arma::colvec operator()(_RandomNumberEngine engine) const {
        arma::colvec p(parameters_.n_elem);
        double sum = 0;
        for (unsigned int j = 0; j < parameters_.n_elem; ++j) {
            p[j] = std::gamma_distribution<double>(parameters_[j], 1)(engine);
            sum += p[j];
        }
        return p / sum;
    }

private:
    arma::colvec parameters_;
};

}  // namespace ptsm

#endif  // SRC_SAMPLER_DIRICHLET_HPP_
