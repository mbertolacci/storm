#ifndef SRC_INDEPENDENT_SAMPLER_MEAT_HPP_
#define SRC_INDEPENDENT_SAMPLER_MEAT_HPP_

#include "sampler/categorical.hpp"
#include "sampler/dirichlet.hpp"

namespace ptsm {

template<typename _RandomNumberEngine>
IndependentSampler::Sample IndependentSampler::operator()(_RandomNumberEngine engine) {
    // Update yMissing

    // Sample p
    arma::colvec counts(pPrior_.n_elem, arma::fill::zeros);
    for (unsigned int i = 0; i < current_.z.n_elem; ++i) {
        ++counts[current_.z[i] - 1];
    }
    current_.p = ptsm::DirichletSampler(counts + pPrior_)(engine);

    // // Sample distribution parameters
    // for (unsigned int k = 0; k < distributions_.size(); ++k) {
    //     arma::ucolvec indices = arma::find(current_.z == k + 2);
    //     current_.distribution[k] = distributionSamplers_[k](engine, data_[indices]);
    // }

    // Sample z
    arma::mat p(data_.n_elem, distributions_.size());
    // for (unsigned int k = 0; k < distributions_.size(); ++k) {
    //     p.col(k).fill(current_.p[k + 1]);
    // }

    for (unsigned int i = 0; i < current_.z.n_elem; ++i) {
        current_.z[i] = ptsm::CategoricalSampler(p.row(i))(engine);
    }

    return current_;
}

template<typename _RandomNumberEngine>
IndependentSampler::Samples IndependentSampler::operator()(
    _RandomNumberEngine engine,
    unsigned int nSamples, unsigned int burnIn,
    unsigned int distributionThin, unsigned int pThin,
    unsigned int zThin, unsigned int yMissingThin
) {
    return IndependentSampler::Samples();
}

}  // namespace ptsm

#endif  // SRC_INDEPENDENT_SAMPLER_MEAT_HPP_
