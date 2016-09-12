#include <RcppArmadillo.h>
#include "rng.hpp"

using arma::colvec;
using arma::diagmat;
using arma::mat;

using Rcpp::as;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

using ptsm::RNG;
using ptsm::rng;

class GPSampler {
 public:
    GPSampler(
        NumericVector y,
        NumericMatrix designMatrix,
        NumericVector betaPriorMean,
        NumericVector betaPriorVariance,
        double variancePriorAlpha,
        double variancePriorBeta,
        double tauSquaredPriorAlpha,
        double tauSquaredPriorBeta,
        unsigned int nGPBases
    ) : variancePriorAlpha_(variancePriorAlpha),
        variancePriorBeta_(variancePriorBeta),
        tauSquaredPriorAlpha_(tauSquaredPriorAlpha),
        tauSquaredPriorBeta_(tauSquaredPriorBeta),
        nGPBases_(nGPBases) {
        // Data
        y_ = as<colvec>(y);
        designMatrix_ = as<mat>(designMatrix);
        // Precompute
        XtX_ = designMatrix_.t() * designMatrix_;

        // Priors
        betaPriorMean_ = as<colvec>(betaPriorMean);
        betaPriorVariance_ = as<colvec>(betaPriorVariance);

        // Starting values
        betaCurrent_ = colvec(0.01, nGPBases_);
        varianceCurrent_ = 1;
        tauSquaredCurrent_ = 1;
    }

    void start() {};
    void next();

    const colvec getBetaCurrent() {
        return betaCurrent_;
    }

    const double getVarianceCurrent() {
        return varianceCurrent_;
    }

    const double getTauSquaredCurrent() {
        return tauSquaredCurrent_;
    }

 protected:
    // Configuration
    mat designMatrix_;

    // Data
    colvec y_;

    // Priors
    colvec betaPriorMean_;
    colvec betaPriorVariance_;
    double variancePriorAlpha_;
    double variancePriorBeta_;
    double tauSquaredPriorAlpha_;
    double tauSquaredPriorBeta_;
    unsigned int nGPBases_;

    // Current sampler state
    colvec betaCurrent_;
    double varianceCurrent_;
    double tauSquaredCurrent_;

    // Precomputed
    mat XtX_;

    void sampleBeta_();
    void sampleVariance_();
    void sampleTauSquared();
};

void GPSampler::next() {
    // Sample \bm{\tau}^2
    sampleTauSquared();

    // Sample \bm{\sigma}^2
    sampleVariance_();

    // Sample /bm{\beta}
    sampleBeta_();
}

void GPSampler::sampleBeta_() {
    mat currentPriorPrecision = diagmat(1 / betaPriorVariance_);

    // R'R = V^{-1} = \sigma^{-2} X' X + \Sigma_0^{-1}
    mat R = chol(XtX_ / varianceCurrent_ + currentPriorPrecision);
    // R'z = \sigma^{-2} X'y + \Sigma_0^{-1} \beta_0
    mat z = solve(R.t(), designMatrix_.t() * y_ / varianceCurrent_ + currentPriorPrecision * betaPriorMean_);
    // R \beta = z
    colvec betaHat = solve(R, z);

    betaCurrent_ = betaHat + solve(R, rng.randn(designMatrix_.n_cols));
}

void GPSampler::sampleVariance_() {
    colvec residuals = y_ - designMatrix_ * betaCurrent_;
    varianceCurrent_ = 1 / rng.randg(
        variancePriorAlpha_ + static_cast<double>(y_.n_elem) / 2,
        1 / (variancePriorBeta_ + arma::dot(residuals, residuals) / 2)
    );
}

void GPSampler::sampleTauSquared() {
    double nBases = nGPBases_;
    colvec betaGP = betaCurrent_.tail(nGPBases_);
    tauSquaredCurrent_ = 1 / rng.randg(
        tauSquaredPriorAlpha_ + nBases / 2,
        1 / (tauSquaredPriorBeta_ + arma::dot(betaGP, betaGP) / 2)
    );
    betaPriorVariance_.tail(nGPBases_) = tauSquaredCurrent_ * arma::ones(nGPBases_);
}


// [[Rcpp::export(name=".ptsm_gp_sample")]]
List gpSample(
    NumericVector y,
    NumericMatrix designMatrix,
    NumericVector betaPriorMean,
    NumericVector betaPriorVariance,
    double variancePriorAlpha,
    double variancePriorBeta,
    double tauSquaredPriorAlpha,
    double tauSquaredPriorBeta,
    unsigned int nGPBases
) {
    RNG::initialise();

    GPSampler sampler(
        y,
        designMatrix,
        betaPriorMean,
        betaPriorVariance,
        variancePriorAlpha,
        variancePriorBeta,
        tauSquaredPriorAlpha,
        tauSquaredPriorBeta,
        nGPBases
    );
    sampler.start();

    List output;
    return output;
}