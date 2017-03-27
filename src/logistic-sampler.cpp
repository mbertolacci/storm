#include <RcppArmadillo.h>
#include <vector>

#include "logistic-sampler.hpp"
#include "logistic-utils.hpp"
#include "utils.hpp"

using arma::colvec;
using arma::cube;
using arma::diagmat;
using arma::field;
using arma::mat;
using arma::rowvec;
using arma::ucolvec;

using Rcpp::as;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;
using Rcpp::StringVector;

unsigned int sampleSingleZ3(const colvec &p) {
    double u = rng.randu();
    colvec pNormalised = p / arma::sum(p);
    for (unsigned int k = 0; k < pNormalised.n_elem; ++k) {
        if (u < pNormalised[k]) {
            return k + 1;
        }
        u -= pNormalised[k];
    }
    return pNormalised.n_elem - 1;
}

cube getLogisticPTransition(const mat delta, const mat designMatrixBase) {
    unsigned int nStates = delta.n_rows + 1;
    mat designMatrix(designMatrixBase);
    cube pTransition(designMatrix.n_rows, nStates, nStates);
    unsigned int nDeltas = designMatrix.n_cols;

    for (unsigned int fromState = 0; fromState < nStates; ++fromState) {
        for (unsigned int k = 0; k < nStates - 1; ++k) {
            designMatrix.col(nDeltas + k - nStates + 1).fill(0);
        }
        if (fromState > 0) {
            designMatrix.col(nDeltas + fromState - nStates).fill(1);
        }
        pTransition.slice(fromState).cols(1, nStates - 1) = getLogisticP(delta, designMatrix);
        pTransition.slice(fromState).col(0) = 1.0 - arma::sum(pTransition.slice(fromState).cols(1, nStates - 1), 1);
    }

    return pTransition;
}

LogisticSampler::LogisticSampler(
    List panelY, List panelDesignMatrix, unsigned int order,
    StringVector distributionNames, List priors, List samplingSchemes,
    List panelZStart, IntegerVector panelZ0Start, List thetaStart,
    List panelDeltaStart, NumericVector deltaFamilyMeanStart, NumericMatrix deltaFamilyVarianceStart,
    Rcpp::Nullable<NumericMatrix> deltaFamilyDesignMatrix
) : order_(order),
    nDataLevels_(panelDesignMatrix.length()),
    logger_("ptsm.logistic_sample") {
    panelDesignMatrix_ = fieldFromList<mat>(panelDesignMatrix);
    unsigned int nComponents = distributionNames.length();
    if (order > 0) {
        // Add room for the latent variable lags
        for (unsigned int level = 0; level < nDataLevels_; ++level) {
            panelDesignMatrix_[level].resize(
                panelDesignMatrix_[level].n_rows,
                panelDesignMatrix_[level].n_cols + nComponents * order
            );
        }
    }

    nDeltas_ = panelDesignMatrix_[0].n_cols;

    // Priors and samplers
    List distributionsPrior = priors["distributions"];
    List distributionsSamplingSchemes = samplingSchemes["distributions"];
    for (unsigned int k = 0; k < nComponents; ++k) {
        distributions_.push_back(Distribution(distributionNames[k]));
        distributionSamplers_.push_back(ParameterSampler(
            distributionsPrior[k], distributionsSamplingSchemes[k],
            distributions_[k]
        ));
    }

    List logisticPriorConfiguration = priors["logistic"];
    logisticParameterHierarchical_ = strcmp(logisticPriorConfiguration["type"], "hierarchical") == 0;
    if (logisticPriorConfiguration.containsElementNamed("is_gp")) {
        logisticParameterGaussianProcess_ = static_cast<int>(logisticPriorConfiguration["is_gp"]);
    } else {
        logisticParameterGaussianProcess_ = false;
    }

    if (logisticParameterGaussianProcess_) {
        nGPBases_ = static_cast<unsigned int>(logisticPriorConfiguration["n_gp_bases"]);
    }

    if (logisticParameterHierarchical_) {
        deltaFamilyDesignMatrix_ = as<mat>(deltaFamilyDesignMatrix);
        UtU_ = deltaFamilyDesignMatrix_.t() * deltaFamilyDesignMatrix_;

        List logisticFamilyMeanPrior = logisticPriorConfiguration["mean"];
        familyMeanPriorMean_ = as<cube>(logisticFamilyMeanPrior["mean"]);
        familyMeanPriorVariance_ = as<cube>(logisticFamilyMeanPrior["variance"]);

        List logisticFamilyVariancePrior = logisticPriorConfiguration["variance"];
        familyVariancePriorAlpha_ = as<mat>(logisticFamilyVariancePrior["alpha"]);
        familyVariancePriorBeta_ = as<mat>(logisticFamilyVariancePrior["beta"]);

        if (logisticParameterGaussianProcess_) {
            List logisticFamilyTauSquaredPrior = logisticPriorConfiguration["tau_squared"];
            familyTauSquaredPriorAlpha_ = as<mat>(logisticFamilyTauSquaredPrior["alpha"]);
            familyTauSquaredPriorBeta_ = as<mat>(logisticFamilyTauSquaredPrior["beta"]);
        }
        nLevels_ = deltaFamilyDesignMatrix_.n_rows;
        nMissingLevels_ = nLevels_ - nDataLevels_;
    } else {
        nLevels_ = nDataLevels_;
        nMissingLevels_ = 0;
    }

    logisticParameterPrior_ = LogisticParameterPrior(logisticPriorConfiguration);
    for (unsigned int level = 0; level < nDataLevels_; ++level) {
        panelLogisticParameterSampler_.push_back(
            LogisticParameterSampler(as<List>(as<List>(samplingSchemes["logistic"])[level]))
        );
    }

    // Data
    panelYCurrent_ = fieldFromList<colvec>(panelY);
    panelLogYCurrent_ = field<colvec>(nDataLevels_);
    for (unsigned int level = 0; level < nDataLevels_; ++level) {
        panelLogYCurrent_[level] = log(panelYCurrent_[level]);
    }
    panelYMissingIndices_ = field<ucolvec>(nDataLevels_);
    panelYIsMissing_ = std::vector< std::vector<bool> >(nDataLevels_);
    for (unsigned int level = 0; level < nDataLevels_; ++level) {
        MissingValuesPair missingValuesPair = findMissingValues(panelY[level]);
        panelYMissingIndices_[level] = missingValuesPair.first;
        panelYIsMissing_[level] = missingValuesPair.second;
    }

    // Starting values
    panelZ0Current_ = as<ucolvec>(panelZ0Start);
    panelZCurrent_ = fieldFromList<ucolvec>(panelZStart);
    distributionCurrent_ = fieldFromList<colvec>(thetaStart);
    panelDeltaCurrent_ = cubeFromList(panelDeltaStart);
    deltaFamilyMeanCurrent_ = as<cube>(deltaFamilyMeanStart);
    deltaFamilyVarianceCurrent_ = as<mat>(deltaFamilyVarianceStart);
    deltaFamilyTauSquaredCurrent_ = mat(panelDeltaCurrent_.n_rows, nDeltas_);
}

void LogisticSampler::start() {
    #pragma omp parallel for
    for (unsigned int level = 0; level < nDataLevels_; ++level) {
        sampleMissingY(
            panelYCurrent_[level], panelLogYCurrent_[level], panelYMissingIndices_[level], panelZCurrent_[level],
            getParameterBoundDistributions()
        );

        if (order_ > 0) {
            unsigned int nComponents = distributions_.size();
            for (unsigned int k = 0; k < nComponents; ++k) {
                panelDesignMatrix_[level](0, nDeltas_ + k - nComponents) = panelZ0Current_[level] == k + 2;
                for (unsigned int i = 0; i < panelZCurrent_[level].n_elem - 1; ++i) {
                    panelDesignMatrix_[level](i + 1, nDeltas_ + k - nComponents) = panelZCurrent_[level][i] == k + 2;
                }
            }
        }
    }
}

void LogisticSampler::next() {
    if (logisticParameterHierarchical_) {
        if (logisticParameterGaussianProcess_) {
            // Sample \bm{\tau}^2
            sampleDeltaFamilyTau_();
        }

        // Sample \bm{\sigma}^2
        sampleDeltaFamilyVariance_();

        // Sample /bm{\mu}
        sampleDeltaFamilyMean_();
    }

    // Sample the parameters of the mixture distributions
    sampleDistributions_();

    #pragma omp parallel for
    for (unsigned int level = 0; level < nDataLevels_; ++level) {
        sampleLevel_(level);
    }

    #pragma omp parallel for
    for (unsigned int level = nDataLevels_; level < nLevels_; ++level) {
        sampleMissingLevel_(level);
    }
}

void LogisticSampler::sampleDeltaFamilyMean_() {
    #pragma omp parallel for collapse(2)
    for (unsigned int i = 0; i < panelDeltaCurrent_.n_rows; ++i) {
        for (unsigned int j = 0; j < panelDeltaCurrent_.n_cols; ++j) {
            colvec delta = panelDeltaCurrent_.tube(i, j);
            double variance = deltaFamilyVarianceCurrent_(i, j);

            colvec currentPriorMean = familyMeanPriorMean_.tube(i, j);
            colvec currentPriorVariance = familyMeanPriorVariance_.tube(i, j);
            mat currentPriorPrecision = diagmat(1 / currentPriorVariance);

            // R'R = V^{-1} = \sigma^{-2} U' U + \Sigma_0^{-1}
            mat R = chol(UtU_ / variance + currentPriorPrecision);
            // R'z = \sigma^{-2} U'\delta + \Sigma_0^{-1} \beta_0
            mat z = solve(trimatl(R.t()), deltaFamilyDesignMatrix_.t() * delta / variance + currentPriorPrecision * currentPriorMean);
            // R \beta = z
            colvec betaHat = solve(trimatu(R), z);

            deltaFamilyMeanCurrent_.tube(i, j) = betaHat + solve(trimatu(R), rng.randn(deltaFamilyDesignMatrix_.n_cols));
        }
    }
}

void LogisticSampler::sampleDeltaFamilyVariance_() {
    double nLevels = panelDeltaCurrent_.n_slices;

    #pragma omp parallel for collapse(2)
    for (unsigned int i = 0; i < panelDeltaCurrent_.n_rows; ++i) {
        for (unsigned int j = 0; j < panelDeltaCurrent_.n_cols; ++j) {
            colvec delta = panelDeltaCurrent_.tube(i, j);
            colvec beta = deltaFamilyMeanCurrent_.tube(i, j);

            colvec residuals = delta - deltaFamilyDesignMatrix_ * beta;

            deltaFamilyVarianceCurrent_(i, j) = 1 / rng.randg(
                familyVariancePriorAlpha_(i, j) + nLevels / 2,
                1 / (familyVariancePriorBeta_(i, j) + arma::dot(residuals, residuals) / 2)
            );
        }
    }
}

void LogisticSampler::sampleDeltaFamilyTau_() {
    double nBases = nGPBases_;

    #pragma omp parallel for
    for (unsigned int i = 0; i < panelDeltaCurrent_.n_rows; ++i) {
        for (unsigned int j = 0; j < panelDeltaCurrent_.n_cols; ++j) {
            colvec betaGP = deltaFamilyMeanCurrent_.tube(i, j);
            betaGP = betaGP.tail(nGPBases_);

            deltaFamilyTauSquaredCurrent_(i, j) = 1 / rng.randg(
                familyTauSquaredPriorAlpha_(i, j) + nBases / 2,
                1 / (familyTauSquaredPriorBeta_(i, j) + arma::dot(betaGP, betaGP) / 2)
            );

            familyMeanPriorVariance_.subcube(
                i, j, familyMeanPriorVariance_.n_slices - nGPBases_,
                i, j, familyMeanPriorVariance_.n_slices - 1
            ) = deltaFamilyTauSquaredCurrent_(i, j) * arma::ones(nGPBases_);
        }
    }
}

void LogisticSampler::sampleDistributions_() {
    colvec y = vectorise(panelYCurrent_);
    colvec logY = vectorise(panelLogYCurrent_);
    ucolvec zCurrent = vectorise(panelZCurrent_);

    for (unsigned int k = 0; k < distributions_.size(); ++k) {
        distributionCurrent_[k] = distributionSamplers_[k].sample(
            distributionCurrent_[k],
            DataBoundDistribution(y, logY, zCurrent, k + 2, distributions_[k])
        );
    }
}

LogisticParameterPrior LogisticSampler::getLevelLogisticPrior_(unsigned int level) {
    mat deltaLevelMean(size(deltaFamilyVarianceCurrent_));
    if (logisticParameterHierarchical_) {
        rowvec deltaDesignLevelRow = deltaFamilyDesignMatrix_.row(level);
        for (unsigned int i = 0; i < deltaLevelMean.n_rows; ++i) {
            for (unsigned int j = 0; j < deltaLevelMean.n_cols; ++j) {
                colvec deltaFamilyMeans = deltaFamilyMeanCurrent_.tube(i, j);
                deltaLevelMean(i, j) = as_scalar(deltaDesignLevelRow * deltaFamilyMeans);
            }
        }
    } else {
        deltaLevelMean = deltaFamilyMeanCurrent_.slice(0);
    }

    return logisticParameterPrior_.withParameters(
        deltaLevelMean,
        deltaFamilyVarianceCurrent_
    );
}

void LogisticSampler::sampleLevel_(unsigned int level) {
    // Sample \bm{z}
    auto boundDistributions = getParameterBoundDistributions();
    if (order_ == 0) {
        auto pCurrent = getLogisticP(panelDeltaCurrent_.slice(level), panelDesignMatrix_[level]);
        panelZ0Current_[level] = sampleSingleZ(pCurrent.row(0).t());
        panelZCurrent_[level] = sampleZ(
            pCurrent, panelYCurrent_[level], panelYIsMissing_[level],
            boundDistributions
        );
    } else {
        auto pTransition = getLogisticPTransition(panelDeltaCurrent_.slice(level), panelDesignMatrix_[level]);
        unsigned int nStates = distributions_.size() + 1;
        mat pMarginal(panelYCurrent_[level].n_elem, nStates, arma::fill::zeros);

        for (unsigned int i = 0; i < panelYCurrent_[level].n_elem; ++i) {
            if (panelYCurrent_[level][i] == 0 && !panelYIsMissing_[level][i]) {
                pMarginal(i, 0) = 1;
            } else {
                for (unsigned int toState = 0; toState < nStates; ++toState) {
                    for (unsigned int fromState = 0; fromState < nStates; ++fromState) {
                        if (i == 0) {
                            pMarginal(i, toState) += pTransition(i, toState, fromState) / static_cast<double>(nStates);
                        } else {
                            pMarginal(i, toState) += pTransition(i, toState, fromState) * pMarginal(i - 1, fromState);
                        }
                    }
                }
                if (!panelYIsMissing_[level][i]) {
                    // We know at this point that y is not 0
                    pMarginal(i, 0) = 0;
                    for (unsigned int state = 0; state < distributions_.size(); ++state) {
                        pMarginal(i, state + 1) *= boundDistributions[state].pdf(panelYCurrent_[level][i]);
                    }
                    pMarginal.row(i) /= arma::sum(pMarginal.row(i));
                }
            }
        }
        unsigned int lastIndex = panelZCurrent_[level].n_elem - 1;
        panelZCurrent_[level][lastIndex] = sampleSingleZ3(pMarginal.row(lastIndex).t());
        for (int i = lastIndex - 1; i >= 0; --i) {
            colvec pNext = pTransition.tube(i + 1, panelZCurrent_[level][i + 1] - 1);
            panelZCurrent_[level][i] = sampleSingleZ3(
                pMarginal.row(i).t() % pNext
            );
        }
        colvec pFirst = pTransition.tube(0, panelZCurrent_[level][0] - 1);
        panelZ0Current_[level] = sampleSingleZ3(
            pFirst / static_cast<double>(nStates)
        );
    }

    // Sample missing values \bm{y^*}
    sampleMissingY(
        panelYCurrent_[level], panelLogYCurrent_[level], panelYMissingIndices_[level], panelZCurrent_[level],
        boundDistributions
    );

    if (order_ > 0) {
        unsigned int nComponents = distributions_.size();
        for (unsigned int k = 0; k < nComponents; ++k) {
            panelDesignMatrix_[level](0, nDeltas_ + k - nComponents) = panelZ0Current_[level] == k + 2;
            for (unsigned int i = 0; i < panelZCurrent_[level].n_elem - 1; ++i) {
                panelDesignMatrix_[level](i + 1, nDeltas_ + k - nComponents) = panelZCurrent_[level][i] == k + 2;
            }
        }
    }

    // Sample \Delta_s
    LogisticParameterPrior boundLogisticParameterPrior = getLevelLogisticPrior_(level);
    panelDeltaCurrent_.slice(level) = panelLogisticParameterSampler_[level].sample(
        panelDeltaCurrent_.slice(level),
        panelZCurrent_[level],
        panelDesignMatrix_[level],
        boundLogisticParameterPrior
    );
}

void LogisticSampler::sampleMissingLevel_(unsigned int level) {
    LogisticParameterPrior boundLogisticParameterPrior = getLevelLogisticPrior_(level);
    panelDeltaCurrent_.slice(level) = boundLogisticParameterPrior.sample();
}

std::vector<ParameterBoundDistribution> LogisticSampler::getParameterBoundDistributions() {
    std::vector<ParameterBoundDistribution> output;
    for (unsigned int k = 0; k < distributions_.size(); ++k) {
        output.push_back(ParameterBoundDistribution(
            distributionCurrent_[k],
            distributions_[k]
        ));
    }
    return output;
}

