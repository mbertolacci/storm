#include <Rcpp.h>

class ThetaValues {
public:
    double alpha;
    int alphaAccept;
    double beta;
    int betaAccept;
    double gammaLogDensity;

    double mu;
    int muAccept;
    double sigma;
    int sigmaAccept;
    double xi;
    int xiAccept;
    double gevLogDensity;

    ThetaValues()
        : alphaAccept{0},
          betaAccept{0},
          muAccept{0},
          sigmaAccept{0},
          xiAccept{0} {}

    ThetaValues(Rcpp::NumericVector initialValues)
        : alphaAccept{0},
          betaAccept{0},
          muAccept{0},
          sigmaAccept{0},
          xiAccept{0},
          alpha{initialValues[0]},
          beta{initialValues[1]},
          mu{initialValues[2]},
          sigma{initialValues[3]},
          xi{initialValues[4]} {}

    bool isInGEVSupport(double y);
    void printGEVSupport();
    void printAcceptanceRatios(int nIterations);
    void resetAcceptCounts();
};

double gammaLogDensity(ThetaValues theta, double sumGammaY, double sumLogGammaY, int nGamma);
double gevLogDensity(ThetaValues theta, Rcpp::NumericVector y, Rcpp::NumericVector zCurrent);

ThetaValues sampleJoint(ThetaValues thetaCurrent, Rcpp::NumericVector y, double sumGammaY, double sumLogGammaY, int nGamma, Rcpp::NumericVector zCurrent, Rcpp::NumericMatrix jointCholeskyPrior);

ThetaValues sampleAlpha(ThetaValues thetaCurrent, double sumGammaY, double sumLogGammaY, int nGamma, double jumpSigma);
ThetaValues sampleBeta(ThetaValues thetaCurrent, double sumGammaY, double sumLogGammaY, int nGamma, double jumpSigma);
ThetaValues sampleXi(ThetaValues thetaCurrent, Rcpp::NumericVector y, Rcpp::NumericVector zCurrent, double jumpSigma);
ThetaValues sampleSigma(ThetaValues thetaCurrent, Rcpp::NumericVector y, Rcpp::NumericVector zCurrent, double jumpSigma);
ThetaValues sampleMu(ThetaValues thetaCurrent, Rcpp::NumericVector y, Rcpp::NumericVector zCurrent, double jumpSigma);
ThetaValues sampleMuAndSigma(
    ThetaValues thetaCurrent, Rcpp::NumericVector y, Rcpp::NumericVector zCurrent,
    double muJumpSigma, double sigmaJumpSigma, double rho
);
double sampleYGivenZ(ThetaValues thetaCurrent, int z);
