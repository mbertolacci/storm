#ifndef SRC_LOGISTIC_SAMPLER_MPI_HPP_
#define SRC_LOGISTIC_SAMPLER_MPI_HPP_

#include <RcppArmadillo.h>
#include "logistic-sampler.hpp"

class LogisticSamplerMPI : public LogisticSampler {
 public:
    using LogisticSampler::LogisticSampler;

    void start();
    void next();

 protected:
    void sampleDistributions_();

 private:
    int rank_;

    arma::icolvec rankCounts_;
    arma::icolvec rankDisplacements_;

    arma::cube panelDeltaCurrentAll_;
    arma::mat deltaFamilyDesignMatrixAll_;

    arma::ucolvec nCurrent_;
    arma::colvec sumYCurrent_;
    arma::colvec sumLogYCurrent_;

    arma::mat UtUAll_;

    void broadcast_();
    void gather_();
};

#endif  // SRC_LOGISTIC_SAMPLER_MPI_HPP_
