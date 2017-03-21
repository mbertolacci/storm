#ifndef SRC_LOGISTIC_SAMPLER_HPP_
#define SRC_LOGISTIC_SAMPLER_HPP_

#include <RcppArmadillo.h>
#include <vector>

#include "distribution.hpp"
#include "hypercube4.hpp"
#include "logistic-parameter-sampler.hpp"
#include "logistic-prior.hpp"
#include "parameter-sampler.hpp"
#include "progress.hpp"
#include "utils.hpp"
#include "logging.hpp"

class LogisticSampler {
 public:
    LogisticSampler(
        Rcpp::List panelY, Rcpp::List panelDesignMatrix, unsigned int order,
        Rcpp::StringVector distributionNames, Rcpp::List priors, Rcpp::List samplingSchemes,
        Rcpp::List panelZStart, Rcpp::IntegerVector panelZ0Start, Rcpp::List thetaStart,
        Rcpp::List panelDeltaStart, Rcpp::NumericVector deltaFamilyMeanStart,
        Rcpp::NumericMatrix deltaFamilyVarianceStart,
        Rcpp::Nullable<Rcpp::NumericMatrix> deltaDesignMatrix
    );

    void start();
    void next();

    bool getIsHierarchical() {
        return logisticParameterHierarchical_;
    }

    bool getIsGaussianProcess() {
        return logisticParameterGaussianProcess_;
    }

    const unsigned int getNComponents() {
        return distributions_.size();
    }

    const arma::colvec getDistributionCurrent(unsigned int distribution) {
        return distributionCurrent_[distribution];
    }

    const arma::cube getPanelDeltaCurrent() {
        return panelDeltaCurrent_;
    }

    const arma::cube getDeltaFamilyMeanCurrent() {
        return deltaFamilyMeanCurrent_;
    }

    const arma::mat getDeltaFamilyVarianceCurrent() {
        return deltaFamilyVarianceCurrent_;
    }

    const arma::mat getDeltaFamilyTauSquaredCurrent() {
        return deltaFamilyTauSquaredCurrent_;
    }

    const arma::ucolvec getPanelZ0Current() {
        return panelZ0Current_;
    }

    const arma::ucolvec getPanelZCurrent(unsigned int level) {
        return panelZCurrent_[level];
    }

    const arma::colvec getPanelYMissingCurrent(unsigned int level) {
        return panelYCurrent_[level](panelYMissingIndices_[level]);
    }

 protected:
    // Configuration
    unsigned int order_;

    arma::field<arma::mat> panelDesignMatrix_;
    arma::mat deltaFamilyDesignMatrix_;

    std::vector<Distribution> distributions_;

    arma::field<arma::ucolvec> panelYMissingIndices_;
    std::vector< std::vector<bool> > panelYIsMissing_;

    // Priors
    bool logisticParameterHierarchical_;
    bool logisticParameterGaussianProcess_;
    LogisticParameterPrior logisticParameterPrior_;
    arma::cube familyMeanPriorMean_;
    arma::cube familyMeanPriorVariance_;
    arma::mat familyVariancePriorAlpha_;
    arma::mat familyVariancePriorBeta_;
    arma::mat familyTauSquaredPriorAlpha_;
    arma::mat familyTauSquaredPriorBeta_;
    unsigned int nGPBases_;

    // Samplers
    std::vector<ParameterSampler> distributionSamplers_;
    std::vector<LogisticParameterSampler> panelLogisticParameterSampler_;

    // Easy accessors
    unsigned int nDataLevels_;
    unsigned int nMissingLevels_;
    unsigned int nLevels_;
    unsigned int nDeltas_;

    // Current sampler state
    // NOTE(mgnb): these fall here, in state, because if there are missing values they are filled in here
    arma::field<arma::colvec> panelYCurrent_;
    arma::ucolvec panelZ0Current_;
    arma::field<arma::ucolvec> panelZCurrent_;
    arma::cube panelDeltaCurrent_;

    arma::field<arma::colvec> distributionCurrent_;

    arma::cube deltaFamilyMeanCurrent_;
    arma::mat deltaFamilyVarianceCurrent_;
    // Used for Gaussian Process (GP) prior
    arma::mat deltaFamilyTauSquaredCurrent_;

    // Computed from current state
    arma::field<arma::colvec> panelLogYCurrent_;
    arma::mat UtU_;

    ptsm::Logger logger_;

    void sampleDeltaFamilyMean_();
    void sampleDeltaFamilyVariance_();
    void sampleDeltaFamilyTau_();
    void sampleDistributions_();
    LogisticParameterPrior getLevelLogisicPrior_(unsigned int level);
    void sampleLevel_(unsigned int level);
    void sampleMissingLevel_(unsigned int level);

    std::vector<ParameterBoundDistribution> getParameterBoundDistributions();
};

template <typename T>
class LogisticSample {
 public:
    LogisticSample(
        T& sampler,
        unsigned int nSamples, unsigned int burnIn,
        unsigned int distributionThinning, unsigned int deltaThinning, unsigned int familyThinning,
        unsigned int z0Thinning, unsigned int zThinning, unsigned int yMissingThinning,
        bool progress = false, unsigned int check_interrupt_interval = 10
    ) {
        unsigned int nIterations = nSamples + burnIn;
        unsigned int nDataLevels = sampler.getPanelZ0Current().n_elem;
        unsigned int nComponents = sampler.getNComponents();

        if (distributionThinning > 0) {
            unsigned int nDistributionSamples = ceil(static_cast<double>(nSamples) / static_cast<double>(distributionThinning));

            distribution_ = arma::field<arma::mat>(nComponents);
            for (unsigned int k = 0; k < nComponents; ++k) {
                distribution_[k] = arma::mat(nDistributionSamples, sampler.getDistributionCurrent(k).n_elem);
            }
        }

        if (deltaThinning > 0) {
            unsigned int nDeltaSamples = ceil(static_cast<double>(nSamples) / static_cast<double>(deltaThinning));

            delta_.set_size(
                sampler.getPanelDeltaCurrent().n_rows, sampler.getPanelDeltaCurrent().n_cols,
                sampler.getPanelDeltaCurrent().n_slices,
                nDeltaSamples
            );
        }

        if (sampler.getIsHierarchical() && familyThinning > 0) {
            unsigned int nFamilySamples = ceil(static_cast<double>(nSamples) / static_cast<double>(familyThinning));

            deltaFamilyMean_.set_size(
                sampler.getDeltaFamilyMeanCurrent().n_rows, sampler.getDeltaFamilyMeanCurrent().n_cols,
                sampler.getDeltaFamilyMeanCurrent().n_slices,
                nFamilySamples
            );
            deltaFamilyVariance_.set_size(
                sampler.getDeltaFamilyVarianceCurrent().n_rows,
                sampler.getDeltaFamilyVarianceCurrent().n_cols,
                nFamilySamples
            );

            if (sampler.getIsGaussianProcess()) {
                deltaFamilyTauSquared_.set_size(
                    sampler.getDeltaFamilyTauSquaredCurrent().n_rows,
                    sampler.getDeltaFamilyTauSquaredCurrent().n_cols,
                    nFamilySamples
                );
            }
        }

        if (z0Thinning > 0) {
            unsigned int nZ0Samples = ceil(static_cast<double>(nSamples) / static_cast<double>(z0Thinning));
            panelZ0_ = arma::umat(nZ0Samples, nDataLevels);
        }

        if (zThinning > 0) {
            unsigned int nZSamples = ceil(static_cast<double>(nSamples) / static_cast<double>(zThinning));
            panelZ_ = arma::field<arma::umat>(nDataLevels);
            for (unsigned int level = 0; level < nDataLevels; ++level) {
                panelZ_[level] = arma::umat(nZSamples, sampler.getPanelZCurrent(level).n_elem);
            }
        }

        if (yMissingThinning > 0) {
            unsigned int nYMissingSamples = ceil(static_cast<double>(nSamples) / static_cast<double>(yMissingThinning));
            panelYMissing_ = arma::field<arma::mat>(nDataLevels);
            for (unsigned int level = 0; level < nDataLevels; ++level) {
                panelYMissing_[level] = arma::mat(nYMissingSamples, sampler.getPanelYMissingCurrent(level).n_elem);
            }
        }

        ProgressBar progressBar(nIterations);
        sampler.start();
        for (unsigned int iteration = 0; iteration < nIterations; ++iteration) {
            sampler.next();

            if (iteration % check_interrupt_interval == 0) {
                // NOTE(mgnb): checks whether the user has pressed Ctrl-C (among other things)
                Rcpp::checkUserInterrupt();
            }

            if (iteration >= burnIn) {
                int index = iteration - burnIn;

                if (distributionThinning > 0 && (index % distributionThinning == 0)) {
                    unsigned int thetaSampleIndex = index / distributionThinning;
                    for (unsigned int k = 0; k < nComponents; ++k) {
                        distribution_[k].row(thetaSampleIndex) = sampler.getDistributionCurrent(k).t();
                    }
                }

                if (deltaThinning > 0 && (index % deltaThinning == 0)) {
                    unsigned int deltaSampleIndex = index / deltaThinning;
                    delta_.set_hyperslice(deltaSampleIndex, sampler.getPanelDeltaCurrent());
                }

                if (familyThinning > 0 && (index % familyThinning == 0) && sampler.getIsHierarchical()) {
                    unsigned int familySampleIndex = index / familyThinning;
                    deltaFamilyMean_.set_hyperslice(familySampleIndex, sampler.getDeltaFamilyMeanCurrent());
                    deltaFamilyVariance_.slice(familySampleIndex) = sampler.getDeltaFamilyVarianceCurrent();

                    if (sampler.getIsGaussianProcess()) {
                        deltaFamilyTauSquared_.slice(familySampleIndex) = sampler.getDeltaFamilyTauSquaredCurrent();
                    }
                }

                if (z0Thinning > 0 && (index % z0Thinning == 0)) {
                    unsigned int z0SampleIndex = index / z0Thinning;
                    panelZ0_.row(z0SampleIndex) = sampler.getPanelZ0Current().t();
                }

                if (zThinning > 0 && (index % zThinning == 0)) {
                    unsigned int zSampleIndex = index / zThinning;
                    for (unsigned int level = 0; level < nDataLevels; ++level) {
                        panelZ_[level].row(zSampleIndex) = sampler.getPanelZCurrent(level).t();
                    }
                }

                if (yMissingThinning > 0 && (index % yMissingThinning == 0)) {
                    int yMissingSampleIndex = index / yMissingThinning;
                    for (unsigned int level = 0; level < nDataLevels; ++level) {
                        panelYMissing_[level].row(yMissingSampleIndex) = sampler.getPanelYMissingCurrent(level).t();
                    }
                }
            }

            if (progress) {
                ++progressBar;
            }
        }
    }

    Rcpp::List asList() {
        Rcpp::List results;

        Rcpp::List sample;
        if (distribution_[0].n_elem > 0) {
            sample["distribution"] = listFromField(distribution_);
        }
        if (delta_.n_elem > 0) {
            sample["delta"] = delta_.asNumericVector();
        }
        if (deltaFamilyMean_.n_elem > 0) {
            sample["delta_family_mean"] = deltaFamilyMean_.asNumericVector();
        }
        if (deltaFamilyVariance_.n_elem > 0) {
            sample["delta_family_variance"] = Rcpp::wrap(deltaFamilyVariance_);
        }
        if (deltaFamilyTauSquared_.n_elem > 0) {
            sample["delta_family_tau_squared"] = Rcpp::wrap(deltaFamilyTauSquared_);
        }
        if (panelZ0_.n_elem > 0) {
            sample["z0"] = Rcpp::wrap(panelZ0_);
        }
        if (panelZ_.n_elem > 0) {
            sample["z"] = listFromField(panelZ_);
        }
        if (panelYMissing_.n_elem > 0) {
            sample["y_missing"] = listFromField(panelYMissing_);
        }
        results["sample"] = sample;

        return results;
    }

 private:
    arma::field<arma::mat> distribution_;
    hypercube4 delta_;
    hypercube4 deltaFamilyMean_;
    arma::cube deltaFamilyVariance_;
    arma::cube deltaFamilyTauSquared_;

    arma::umat panelZ0_;
    arma::field<arma::umat> panelZ_;

    arma::field<arma::mat> panelYMissing_;
};

#endif  // SRC_LOGISTIC_SAMPLER_HPP_
