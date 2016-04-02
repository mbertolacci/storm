#include <Rcpp.h>

#include "threecomponent-mixture-utils.h"

class ParameterSampler {
public:
    enum SamplingScheme {
        ALL_INDEPENDENT,
        GEV_INDEPENDENT,
        GAMMA_INDEPENDENT
    };

    // ParameterSampler(SamplingScheme samplingScheme);

    ThetaValues sample(ThetaValues thetaCurrent, Rcpp::NumericVector y, Rcpp::NumericVector zCurrent);
private:
    SamplingScheme samplingScheme_;

    double alphaSigma_;
    double betaSigma_;

    double muSigma_;
    double sigmaSigma_;
    double xiSigma_;
};
