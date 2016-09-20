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

LogisticSampler::LogisticSampler(
    List panelY, List panelDesignMatrix, unsigned int order,
    StringVector distributionNames, List priors, List samplingSchemes,
    List panelZStart, IntegerVector panelZ0Start, List thetaStart,
    List panelDeltaStart, NumericVector deltaFamilyMeanStart, NumericMatrix deltaFamilyVarianceStart,
    Rcpp::Nullable<NumericMatrix> deltaFamilyDesignMatrix
) : order_(order),
    nLevels_(panelDesignMatrix.length()),
    logger_("ptsm.logistic_sample") {
    panelDesignMatrix_ = fieldFromList<mat>(panelDesignMatrix);
    unsigned int nComponents = distributionNames.length();
    if (order > 0) {
        // Add room for the latent variable lags
        for (unsigned int level = 0; level < nLevels_; ++level) {
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
    }

    logisticParameterPrior_ = LogisticParameterPrior(logisticPriorConfiguration);
    for (unsigned int level = 0; level < nLevels_; ++level) {
        panelLogisticParameterSampler_.push_back(
            LogisticParameterSampler(as<List>(as<List>(samplingSchemes["logistic"])[level]))
        );
    }

    // Data
    panelYCurrent_ = fieldFromList<colvec>(panelY);
    panelLogYCurrent_ = field<colvec>(nLevels_);
    for (unsigned int level = 0; level < nLevels_; ++level) {
        panelLogYCurrent_[level] = log(panelYCurrent_[level]);
    }
    panelYMissingIndices_ = field<ucolvec>(nLevels_);
    panelYIsMissing_ = std::vector< std::vector<bool> >(nLevels_);
    for (unsigned int level = 0; level < nLevels_; ++level) {
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

    panelPCurrent_ = field<mat>(nLevels_);
}

void LogisticSampler::start() {
    #pragma omp parallel for
    for (unsigned int level = 0; level < nLevels_; ++level) {
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

        panelPCurrent_[level] = getLogisticP(panelDeltaCurrent_.slice(level), panelDesignMatrix_[level]);
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

    #pragma omp parallel for schedule(dynamic, 1)
    for (unsigned int level = 0; level < nLevels_; ++level) {
        sampleLevel_(level);
    }

    // Rcpp::Rcout << panelDeltaCurrent_.slice(0) << "\n";
}

void LogisticSampler::sampleDeltaFamilyMean_() {
    mat UtU = deltaFamilyDesignMatrix_.t() * deltaFamilyDesignMatrix_;

    #pragma omp parallel for
    for (unsigned int i = 0; i < panelDeltaCurrent_.n_rows; ++i) {
        for (unsigned int j = 0; j < panelDeltaCurrent_.n_cols; ++j) {
            colvec delta = panelDeltaCurrent_.tube(i, j);
            double variance = deltaFamilyVarianceCurrent_(i, j);

            colvec currentPriorMean = familyMeanPriorMean_.tube(i, j);
            colvec currentPriorVariance = familyMeanPriorVariance_.tube(i, j);
            mat currentPriorPrecision = diagmat(1 / currentPriorVariance);

            // R'R = V^{-1} = \sigma^{-2} U' U + \Sigma_0^{-1}
            mat R = chol(UtU / variance + currentPriorPrecision);
            // R'z = \sigma^{-2} U'\delta + \Sigma_0^{-1} \beta_0
            mat z = solve(R.t(), deltaFamilyDesignMatrix_.t() * delta / variance + currentPriorPrecision * currentPriorMean);
            // R \beta = z
            colvec betaHat = solve(R, z);

            deltaFamilyMeanCurrent_.tube(i, j) = betaHat + solve(R, rng.randn(deltaFamilyDesignMatrix_.n_cols));
        }
    }
}

void LogisticSampler::sampleDeltaFamilyVariance_() {
    double nLevels = panelDeltaCurrent_.n_slices;

    #pragma omp parallel for
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

    {
        ucolvec indices = find(zCurrent == 1);
        Rcpp::Rcout << 1 << " " << indices.n_elem << "\n";
    }

    for (unsigned int k = 0; k < distributions_.size(); ++k) {
        ucolvec indices = find(zCurrent == k + 2);
        Rcpp::Rcout << (k + 2) << " " << indices.n_elem << " " << distributionCurrent_[k].t();
        distributionCurrent_[k] = distributionSamplers_[k].sample(
            distributionCurrent_[k],
            DataBoundDistribution(y, logY, zCurrent, k + 2, distributions_[k])
        );
    }
}

void LogisticSampler::sampleLevel_(unsigned int level) {
    // Sample \bm{z}
    panelZ0Current_[level] = sampleSingleZ(panelPCurrent_[level].row(0).t());
    panelZCurrent_[level] = sampleZ(
        panelPCurrent_[level], panelYCurrent_[level], panelYIsMissing_[level],
        getParameterBoundDistributions()
    );

    // Sample missing values \bm{y^*}
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

    // Sample \Delta_s
    LogisticParameterPrior boundLogisticParameterPrior = logisticParameterPrior_.withParameters(
        deltaLevelMean,
        deltaFamilyVarianceCurrent_
    );
    panelDeltaCurrent_.slice(level) = panelLogisticParameterSampler_[level].sample(
        panelDeltaCurrent_.slice(level),
        panelPCurrent_[level],
        panelZCurrent_[level],
        panelDesignMatrix_[level],
        boundLogisticParameterPrior
    );
    panelPCurrent_[level] = getLogisticP(panelDeltaCurrent_.slice(level), panelDesignMatrix_[level]);
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

