#include <RcppArmadillo.h>
#undef Free

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "logistic-sampler-mpi.hpp"

using arma::colvec;
using arma::cube;
using arma::icolvec;
using arma::mat;
using arma::ucolvec;

#ifdef USE_MPI

void LogisticSamplerMPI::start() {
    nCurrent_ = ucolvec(getNComponents());
    sumYCurrent_ = colvec(getNComponents());
    sumLogYCurrent_ = colvec(getNComponents());

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

    broadcast_();

    int groupSize;
    int nLevels = panelDeltaCurrent_.n_slices;
    if (rank_ == 0) {
        MPI_Comm_size(MPI_COMM_WORLD, &groupSize);

        rankCounts_ = icolvec(groupSize);
        rankDisplacements_ = icolvec(groupSize);

        MPI_Gather(
            &nLevels, 1, MPI_INT,
            rankCounts_.memptr(), 1, MPI_INT,
            0, MPI_COMM_WORLD
        );

        int sumNLevels = 0;
        for (unsigned int i = 0; i < rankCounts_.n_elem; ++i) {
            rankDisplacements_[i] = sumNLevels;
            sumNLevels += rankCounts_[i];
        }

        panelDeltaCurrentAll_ = cube(panelDeltaCurrent_.n_rows, panelDeltaCurrent_.n_cols, sumNLevels);

        // NOTE(mgnb): this is set up as the transpose of the actual matrix because matrices are stored column-wise.
        // This way each machine sends its pieces row-wise, then we transpose later.
        deltaFamilyDesignMatrixAll_ = mat(deltaFamilyDesignMatrix_.n_cols, sumNLevels);
        MPI_Gatherv(
            deltaFamilyDesignMatrix_.t().eval().memptr(), deltaFamilyDesignMatrix_.n_elem, MPI_DOUBLE,
            deltaFamilyDesignMatrixAll_.memptr(),
            (rankCounts_ * deltaFamilyDesignMatrix_.n_cols).eval().memptr(),
            (rankDisplacements_ * deltaFamilyDesignMatrix_.n_cols).eval().memptr(),
            MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );
        deltaFamilyDesignMatrixAll_ = deltaFamilyDesignMatrixAll_.t();
    } else {
        MPI_Gather(
            &nLevels, 1, MPI_INT,
            NULL, 0, 0,
            0, MPI_COMM_WORLD
        );

        MPI_Gatherv(
            deltaFamilyDesignMatrix_.t().eval().memptr(), deltaFamilyDesignMatrix_.n_elem, MPI_DOUBLE,
            NULL, NULL, NULL, MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );
    }

    LogisticSampler::start();
}

void LogisticSamplerMPI::next() {
    if (rank_ == 0) {
        logger_.trace("Gathering");
    }

    gather_();

    if (rank_ == 0) {
        logger_.trace("Gathered, now sampling");

        std::swap(panelDeltaCurrent_, panelDeltaCurrentAll_);
        std::swap(deltaFamilyDesignMatrix_, deltaFamilyDesignMatrixAll_);

        if (logisticParameterGaussianProcess_) {
            // Sample \bm{\tau}^2
            logger_.trace("Sampling tau");
            sampleDeltaFamilyTau_();
        }
        // Sample \bm{\sigma}^2
        logger_.trace("Sampling delta family variance");
        sampleDeltaFamilyVariance_();
        // Sample /bm{\mu}
        logger_.trace("Sampling delta family mean");
        sampleDeltaFamilyMean_();
        std::swap(panelDeltaCurrent_, panelDeltaCurrentAll_);
        std::swap(deltaFamilyDesignMatrix_, deltaFamilyDesignMatrixAll_);

        // Sample the parameters of the mixture distributions
        logger_.trace("Sampling distributions");
        sampleDistributions_();

        logger_.trace("Sampled, now broadcasting");
    }

    broadcast_();

    if (rank_ == 0) {
        logger_.trace("Broadcasted, now sampling each level");
    }

    #pragma omp parallel for schedule(dynamic, 1)
    for (unsigned int level = 0; level < nLevels_; ++level) {
        sampleLevel_(level);
    }

    if (rank_ == 0) {
        logger_.trace("Sampled each level");
    }
}

void LogisticSamplerMPI::sampleDistributions_() {
    for (unsigned int k = 0; k < distributions_.size(); ++k) {
        Rcpp::Rcout << "Sampling distribution " << k << " " << nCurrent_[k] << " " << sumYCurrent_[k] << " " << sumLogYCurrent_[k] << "\n";
        distributionCurrent_[k] = distributionSamplers_[k].sample(
            distributionCurrent_[k],
            DataBoundDistribution(nCurrent_[k], sumYCurrent_[k], sumLogYCurrent_[k], distributions_[k])
        );
        Rcpp::Rcout << distributionCurrent_[k] << "\n";
    }
}

void LogisticSamplerMPI::broadcast_() {
    for (unsigned int k = 0; k < distributions_.size(); ++k) {
        MPI_Bcast(
            distributionCurrent_[k].memptr(), distributionCurrent_[k].n_elem, MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );
    }
    MPI_Bcast(
        deltaFamilyMeanCurrent_.memptr(), deltaFamilyMeanCurrent_.n_elem, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );
    MPI_Bcast(
        deltaFamilyVarianceCurrent_.memptr(), deltaFamilyVarianceCurrent_.n_elem, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );
}

void LogisticSamplerMPI::gather_() {
    unsigned int nComponents = nCurrent_.n_elem;

    ucolvec n(nComponents, arma::fill::zeros);
    colvec sumY(nComponents, arma::fill::zeros);
    colvec sumLogY(nComponents, arma::fill::zeros);
    for (unsigned int level = 0; level < nLevels_; ++level) {
        for (unsigned int k = 0; k < nComponents; ++k) {
            ucolvec indices = find(panelZCurrent_[level] == k + 2);
            n[k] += indices.n_elem;
            sumY[k] += accu(panelYCurrent_[level](indices));
            sumLogY[k] += accu(panelLogYCurrent_[level](indices));
        }
    }

    if (rank_ == 0) {
        int nPerLevel = panelDeltaCurrent_.n_rows * panelDeltaCurrent_.n_cols;

        MPI_Gatherv(
            panelDeltaCurrent_.memptr(), panelDeltaCurrent_.n_elem, MPI_DOUBLE,
            panelDeltaCurrentAll_.memptr(),
            (rankCounts_ * nPerLevel).eval().memptr(),
            (rankDisplacements_ * nPerLevel).eval().memptr(),
            MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );

        MPI_Reduce(
            n.memptr(), nCurrent_.memptr(), nComponents,
            MPI_INT, MPI_SUM,
            0, MPI_COMM_WORLD
        );
        MPI_Reduce(
            sumY.memptr(), sumYCurrent_.memptr(), nComponents,
            MPI_DOUBLE, MPI_SUM,
            0, MPI_COMM_WORLD
        );
        MPI_Reduce(
            sumLogY.memptr(), sumLogYCurrent_.memptr(), nComponents,
            MPI_DOUBLE, MPI_SUM,
            0, MPI_COMM_WORLD
        );
    } else {
        MPI_Gatherv(
            panelDeltaCurrent_.memptr(), panelDeltaCurrent_.n_elem, MPI_DOUBLE,
            NULL, NULL, NULL, MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );
        MPI_Reduce(
            n.memptr(), NULL, nComponents,
            MPI_INT, MPI_SUM,
            0, MPI_COMM_WORLD
        );
        MPI_Reduce(
            sumY.memptr(), NULL, nComponents,
            MPI_DOUBLE, MPI_SUM,
            0, MPI_COMM_WORLD
        );
        MPI_Reduce(
            sumLogY.memptr(), NULL, nComponents,
            MPI_DOUBLE, MPI_SUM,
            0, MPI_COMM_WORLD
        );
    }
}

#else

void LogisticSamplerMPI::start() {}
void LogisticSamplerMPI::next() {}
void LogisticSamplerMPI::sampleDistributions_() {}
void LogisticSamplerMPI::broadcast_() {}
void LogisticSamplerMPI::gather_() {}

#endif
