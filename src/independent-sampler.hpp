#ifndef SRC_INDEPENDENT_SAMPLER_HPP_
#define SRC_INDEPENDENT_SAMPLER_HPP_

#include <RcppArmadillo.h>
#include <vector>

#include "distribution.hpp"
#include "data.hpp"
#include "parameter-sampler.hpp"
#include "logging.hpp"

namespace ptsm {

class IndependentSampler {
 public:
    struct Sample {
        arma::field<arma::colvec> distribution;
        arma::colvec p;
        arma::colvec z;
        arma::colvec yMissing;
    };

    struct Samples {
        arma::field<arma::mat> distribution;
        arma::mat p;
        arma::mat z;
        arma::mat yMissing;
    };

    IndependentSampler() {};

    template<typename _RandomNumberEngine>
    Sample operator()(_RandomNumberEngine engine);

    template<typename _RandomNumberEngine>
    Samples operator()(
        _RandomNumberEngine engine,
        unsigned int nSamples, unsigned int burnIn,
        unsigned int distributionThin, unsigned int pThin,
        unsigned int zThin, unsigned int yMissingThin
    );

 protected:
    // Configuration
    std::vector<Distribution> distributions_;
    ptsm::Data data_;

    // Samplers
    std::vector<ParameterSampler> distributionSamplers_;

    // Prior
    arma::colvec pPrior_;

    // Current state
    Sample current_;
};

}  // namespace ptsm

#include "independent-sampler-meat.hpp"

#endif  // SRC_INDEPENDENT_SAMPLER_HPP_
