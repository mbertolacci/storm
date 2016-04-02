#include <Rcpp.h>
#include "threecomponent-mixture-utils.h"
#include "gev.h"

using namespace Rcpp;

void ThetaValues::printGEVSupport() {
    Rcout << "GEV support ";
    if (xi == 0) {
        Rcout << "-Inf, Inf\n";
    } else if (xi > 0) {
        Rcout << (mu - sigma / xi) << ", Inf\n";
    } else if (xi < 0) {
        Rcout << "-Inf, " << (mu - sigma / xi) << "\n";
    }
}

void ThetaValues::printAcceptanceRatios(int nIterations) {
    Rcout << "Acceptance ratios\n"
        << "alpha = " << ((double) alphaAccept / (double) nIterations) << "\n"
        << "beta = " << ((double) betaAccept / (double) nIterations) << "\n"
        << "mu = " << ((double) muAccept / (double) nIterations) << "\n"
        << "sigma = " << ((double) sigmaAccept / (double) nIterations) << "\n"
        << "xi = " << ((double) xiAccept / (double) nIterations) << "\n";
}

void ThetaValues::resetAcceptCounts() {
    alphaAccept = 0;
    betaAccept = 0;
    muAccept = 0;
    sigmaAccept = 0;
    xiAccept = 0;
}

bool isInGEVSupport(double y, ThetaValues theta) {
    if (theta.xi == 0) {
        return true;
    } else if (theta.xi < 0) {
        return y < theta.mu - theta.sigma / theta.xi;
    } else {
        return y > theta.mu - theta.sigma / theta.xi;
    }
}

double gammaLogDensity(ThetaValues theta, double sumGammaY, double sumLogGammaY, int nGamma) {
    if (nGamma == 0) {
        return -DBL_MAX;
    } else {
        double logGammaNorm = -lgamma(theta.alpha) - theta.alpha * log(theta.beta);
        return nGamma * logGammaNorm + (theta.alpha - 1) * sumLogGammaY - sumGammaY / theta.beta;
    }
}

double gevLogDensity(ThetaValues theta, NumericVector y, NumericVector zCurrent) {
    double logDensity = 0;
    bool hadData = false;
    int n = y.length();

    #pragma omp parallel for simd reduction(+:logDensity)
    for (int i = 0; i < n; ++i) {
        if (zCurrent[i] == 3) {
            hadData = true;
            logDensity += dgev(y[i], theta.mu, theta.sigma, theta.xi, true);
        }
    }
    if (!hadData) {
        return -DBL_MAX;
    }
    return logDensity;
}




ThetaValues sampleJoint(ThetaValues thetaCurrent, NumericVector y, double sumGammaY, double sumLogGammaY, int nGamma, NumericVector zCurrent, NumericMatrix jointCholeskyPrior) {
    ThetaValues newTheta = thetaCurrent;

    // Ghetto matrix multiply
    NumericVector r(5);
    for (int j = 0; j < 5; ++j) {
        r[j] = R::rnorm(0, 1);
    }

    for (int j = 0; j < 5; ++j) {
        newTheta.alpha += r[j] * jointCholeskyPrior(0, j);
        newTheta.beta += r[j] * jointCholeskyPrior(1, j);
        newTheta.mu += r[j] * jointCholeskyPrior(2, j);
        newTheta.sigma += r[j] * jointCholeskyPrior(3, j);
        newTheta.xi += r[j] * jointCholeskyPrior(4, j);
    }

    if (isInGEVSupport(0, newTheta)) {
        return thetaCurrent;
    }

    newTheta.gevLogDensity = gevLogDensity(newTheta, y, zCurrent);
    newTheta.gammaLogDensity = gammaLogDensity(newTheta, sumGammaY, sumLogGammaY, nGamma);

    double alpha = newTheta.gammaLogDensity + newTheta.gevLogDensity - thetaCurrent.gammaLogDensity - thetaCurrent.gevLogDensity;
    if (log(R::runif(0, 1)) < alpha) {
        newTheta.alphaAccept++;
        newTheta.betaAccept++;
        newTheta.muAccept++;
        newTheta.sigmaAccept++;
        newTheta.xiAccept++;
        return newTheta;
    }
    return thetaCurrent;
}

ThetaValues sampleAlpha(ThetaValues thetaCurrent, double sumGammaY, double sumLogGammaY, int nGamma, double jumpSigma) {
    ThetaValues newTheta = thetaCurrent;
    newTheta.alpha = thetaCurrent.alpha + R::rnorm(0, jumpSigma);
    newTheta.gammaLogDensity = gammaLogDensity(newTheta, sumGammaY, sumLogGammaY, nGamma);

    double alpha = newTheta.gammaLogDensity - thetaCurrent.gammaLogDensity;
    if (log(R::runif(0, 1)) < alpha) {
        newTheta.alphaAccept++;
        return newTheta;
    }
    return thetaCurrent;
}

ThetaValues sampleBeta(ThetaValues thetaCurrent, double sumGammaY, double sumLogGammaY, int nGamma, double jumpSigma) {
    ThetaValues newTheta = thetaCurrent;
    newTheta.beta = thetaCurrent.beta + R::rnorm(0, jumpSigma);
    newTheta.gammaLogDensity = gammaLogDensity(newTheta, sumGammaY, sumLogGammaY, nGamma);

    double alpha = newTheta.gammaLogDensity - thetaCurrent.gammaLogDensity;
    if (log(R::runif(0, 1)) < alpha) {
        newTheta.betaAccept++;
        return newTheta;
    }
    return thetaCurrent;
}

ThetaValues sampleXi(ThetaValues thetaCurrent, NumericVector y, NumericVector zCurrent, double jumpSigma) {
    ThetaValues newTheta = thetaCurrent;

    while (true) {
        newTheta.xi = thetaCurrent.xi + R::rnorm(0, jumpSigma);
        newTheta.gevLogDensity = gevLogDensity(newTheta, y, zCurrent);

        if (!isInGEVSupport(0, newTheta)) {
            break;
            // return thetaCurrent;
        }
    }

    double alpha = newTheta.gevLogDensity - thetaCurrent.gevLogDensity;
    if (log(R::runif(0, 1)) < alpha) {
        newTheta.xiAccept++;
        return newTheta;
    }
    return thetaCurrent;
}

ThetaValues sampleSigma(ThetaValues thetaCurrent, NumericVector y, NumericVector zCurrent, double jumpSigma) {
    ThetaValues newTheta = thetaCurrent;

    while (true) {
        newTheta.sigma = thetaCurrent.sigma + R::rnorm(0, jumpSigma);
        newTheta.gevLogDensity = gevLogDensity(newTheta, y, zCurrent);

        if (!isInGEVSupport(0, newTheta)) {
            break;
            return thetaCurrent;
        }
    }

    double alpha = newTheta.gevLogDensity - thetaCurrent.gevLogDensity;
    if (log(R::runif(0, 1)) < alpha) {
        newTheta.sigmaAccept++;
        return newTheta;
    }
    return thetaCurrent;
}

ThetaValues sampleMu(ThetaValues thetaCurrent, NumericVector y, NumericVector zCurrent, double jumpSigma) {
    ThetaValues newTheta = thetaCurrent;

    while (true) {
        newTheta.mu = thetaCurrent.mu + R::rnorm(0, jumpSigma);
        newTheta.gevLogDensity = gevLogDensity(newTheta, y, zCurrent);

        if (!isInGEVSupport(0, newTheta)) {
            break;
            return thetaCurrent;
        }
    }

    double alpha = newTheta.gevLogDensity - thetaCurrent.gevLogDensity;
    if (log(R::runif(0, 1)) < alpha) {
        newTheta.muAccept++;
        return newTheta;
    }
    return thetaCurrent;
}

ThetaValues sampleMuAndSigma(
    ThetaValues thetaCurrent, NumericVector y, NumericVector zCurrent,
    double muJumpSigma, double sigmaJumpSigma, double rho
) {
    ThetaValues newTheta = thetaCurrent;
    double X = R::rnorm(0, 1);
    double Y = R::rnorm(0, 1);
    newTheta.mu = thetaCurrent.mu + muJumpSigma * X;
    newTheta.sigma = (
        thetaCurrent.sigma
        + sigmaJumpSigma * (rho * X + sqrt(1 - rho * rho) * Y)
    );
    newTheta.gevLogDensity = gevLogDensity(newTheta, y, zCurrent);

    if (isInGEVSupport(0, newTheta)) {
        return thetaCurrent;
    }

    double alpha = newTheta.gevLogDensity - thetaCurrent.gevLogDensity;
    if (log(R::runif(0, 1)) < alpha) {
        newTheta.muAccept++;
        newTheta.sigmaAccept++;
        return newTheta;
    }
    return thetaCurrent;
}

double sampleYGivenZ(ThetaValues thetaCurrent, int z) {
    if (z == 1) {
        return 0;
    } else if (z == 2) {
        return R::rgamma(thetaCurrent.alpha, thetaCurrent.beta);
    } else {
        return rgev(thetaCurrent.mu, thetaCurrent.sigma, thetaCurrent.xi);
    }
}
