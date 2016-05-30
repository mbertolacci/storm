#include <algorithm>
#include "distribution.hpp"

using arma::colvec;
using arma::colvec2;
using arma::mat;
using arma::mat22;
using arma::max;
using arma::ucolvec;

using Rcpp::warning;
using Rcpp::stop;

#pragma omp declare \
    reduction(arma_mat_sum : mat : omp_out += omp_in) \
    initializer (omp_priv(omp_orig))

DataBoundDistribution::DataBoundDistribution(
    colvec y, colvec logY, ucolvec z, unsigned int thisZ, Distribution distribution
) : y_(y),
    logY_(logY),
    z_(z),
    thisZ_(thisZ),
    thisN_(0),
    sumY_(0),
    sumLogY_(0),
    distribution_(distribution) {
    if (distribution_.getType() == GAMMA) {
        for (unsigned int i = 0; i < y_.n_elem; ++i) {
            if (z_[i] == thisZ_) {
                ++thisN_;
                sumY_ += y_[i];
                sumLogY_ += logY_[i];
            }
        }
    } else if (distribution_.getType() == GENERALISED_GAMMA) {
        for (unsigned int i = 0; i < y_.n_elem; ++i) {
            if (z_[i] == thisZ_) {
                ++thisN_;
                sumLogY_ += logY_[i];
            }
        }
    }
}

double DataBoundDistribution::logLikelihood(colvec parameters) const {
    switch (distribution_.getType()) {
    case GAMMA: {
        if (thisN_ == 0) return -DBL_MAX;
        return (
            - thisN_ * parameters[0] * log(parameters[1])
            - thisN_ * lgamma(parameters[0])
            + (parameters[0] - 1) * sumLogY_
            - sumY_ / parameters[1]
        );
        break;
    }
    case GEV: {
        double logDensity = 0;
        bool hadData = false;

        #pragma omp parallel for simd reduction(+:logDensity)
        for (unsigned int i = 0; i < y_.n_elem; ++i) {
            if (z_[i] == thisZ_) {
                hadData = true;
                logDensity += dgev(y_[i], parameters[0], parameters[1], parameters[2], true);
            }
        }
        if (!hadData) {
            return -DBL_MAX;
        }
        return logDensity;
        break;
    }
    case GENERALISED_GAMMA: {
        double mu = parameters[0];
        double sigma = parameters[1];
        double Q = parameters[2];

        if (Q == 0) {
            double sum = 0;
            #pragma omp parallel for simd reduction(+:sum)
            for (unsigned int i = 0; i < logY_.n_elem; ++i) {
                if (z_[i] == thisZ_) {
                    double t = logY_[i] - mu;
                    sum += t * t;
                }
            }

            return (
                thisN_ * (
                    - log(sigma)
                    - log(sqrt(2 * PI))
                )
                - sumLogY_
                - sum / (sigma * sigma)
            );
        } else {
            double invQSquared = 1 / (Q * Q);

            double expSum = 0;
            #pragma omp parallel for simd reduction(+:expSum)
            for (unsigned int i = 0; i < logY_.n_elem; ++i) {
                if (z_[i] == thisZ_) {
                    expSum += exp(Q * (logY_[i] - mu) / sigma);
                }
            }

            return (
                thisN_ * (
                    log(fabs(Q))
                    - log(sigma)
                    + invQSquared * log(invQSquared)
                    - lgamma(invQSquared)
                    - mu / (sigma * Q)
                )
                + (1 / (sigma * Q) - 1) * sumLogY_
                - expSum / (Q * Q)
            );
        }
    }
    default: {
        return -DBL_MAX;
        break;
    }
    }
}

colvec2 DataBoundDistribution::genGammaEllPrime(colvec2 parameters) const {
    double sigma = exp(parameters[0]);
    double Q = parameters[1];
    double QSq = Q * Q;
    double QCb = Q * Q * Q;
    double n = static_cast<double>(thisN_);

    double sumYQSigma = 0;
    double sumYQSigmaLog = 0;
    for (unsigned int i = 0; i < logY_.n_elem; ++i) {
        if (z_[i] == thisZ_) {
            double yQSigma = exp(Q * logY_[i] / sigma);
            sumYQSigma += yQSigma;
            sumYQSigmaLog += yQSigma * logY_[i];
        }
    }

    colvec2 result;
    result[0] = -sigma * Q - sumLogY_ / n + sumYQSigmaLog / sumYQSigma;
    result[1] = (
        1 / Q
        - 2 * log(1 / QSq) / QCb
        + 2 * R::digamma(1 / QSq) / QCb
        - sumLogY_ / (n * sigma * QSq)
        + 2 * log(sumYQSigma / n) / QCb
        - sumYQSigmaLog / (sigma * QSq * sumYQSigma)
    );
    return result;
}

colvec DataBoundDistribution::maximumLikelihoodEstimate(colvec start) const {
    colvec parameters;

    switch (distribution_.getType()) {
    case GAMMA: {
        double s = log(sumY_ / static_cast<double>(thisN_)) - sumLogY_ / static_cast<double>(thisN_);
        double alpha = (3 - s + sqrt((s - 3) * (s - 3) + 24 * s)) / (12 * s);
        double alphaOld;
        int maxIterations = 100;
        int i = 0;
        while (true) {
            alphaOld = alpha;
            alpha = alpha - (log(alpha) - R::digamma(alpha) - s) / ((1 / alpha) - R::trigamma(alpha));
            if (alpha - alphaOld < 0.001) {
                break;
            }
            ++i;
            if (i >= maxIterations) {
                break;
            }
        }

        parameters = colvec(2);
        parameters[0] = alpha;
        parameters[1] = sumY_ / (alpha * thisN_);
        break;
    }
    case GENERALISED_GAMMA: {
        int maxIterations = 100;

        colvec2 thetaCurrent;
        colvec2 thetaOld;
        colvec2 thetaDelta;
        colvec2 ellPrimeCurrent;
        colvec2 ellPrimeOld;
        mat22 mCurrent;
        mat22 mOld;
        colvec2 v;

        mCurrent(0, 0) = 100;
        mCurrent(0, 1) = 0;
        mCurrent(1, 0) = 0;
        mCurrent(1, 1) = 100;
        thetaCurrent[0] = log(start[1]);
        thetaCurrent[1] = start[2];
        ellPrimeCurrent = genGammaEllPrime(thetaCurrent);

        int i = 0;
        while (true) {
            thetaOld = thetaCurrent;
            ellPrimeOld = ellPrimeCurrent;
            mOld = mCurrent;

            thetaCurrent = thetaOld - mOld.i() * ellPrimeOld;
            ellPrimeCurrent = genGammaEllPrime(thetaCurrent);

            thetaDelta = thetaCurrent - thetaOld;
            v = ellPrimeCurrent - ellPrimeOld - mOld * thetaDelta;
            mCurrent = mOld + (v * v.t()) / as_scalar(v.t() * thetaDelta);

            if (max(abs(thetaDelta)) < 0.001) {
                // Rcout << ellPrimeCurrent << "\n";
                break;
            }

            ++i;
            if (i >= maxIterations) {
                break;
            }
        }

        if (i == maxIterations) {
            warning("maximumLikelihoodEstimate failed for gengamma");
            return start;
        }

        parameters = colvec(3);
        parameters[1] = exp(thetaCurrent[0]);
        parameters[2] = thetaCurrent[1];

        double sumYQSigma = 0;
        for (unsigned int i = 0; i < logY_.n_elem; ++i) {
            if (z_[i] == thisZ_) {
                sumYQSigma += exp(parameters[2] * logY_[i] / parameters[1]);
            }
        }
        parameters[0] = (parameters[1] / parameters[2]) * log(sumYQSigma / static_cast<double>(thisN_));

        break;
    }
    default:
        stop("Cannot call maximumLikelihoodEstimate for %s", distribution_.getName());
    }


    return parameters;
}

mat DataBoundDistribution::hessian(colvec parameters) const {
    mat hessian(parameters.n_elem, parameters.n_elem, arma::fill::zeros);

    switch (distribution_.getType()) {
    case GAMMA: {
        double alpha = parameters[0];
        double beta = parameters[1];
        double betaSq = beta * beta;
        double betaCb = beta * beta * beta;

        hessian(0, 0) = -thisN_ * R::trigamma(alpha);
        hessian(0, 1) = -thisN_ / beta;
        hessian(1, 1) = thisN_ * alpha / betaSq - (2 / betaCb) * sumY_;

        break;
    }
    case GEV: {
        double mu = parameters[0];
        double sigma = parameters[1];
        double xi = parameters[2];
        double xiSq = xi * xi;
        double xiCb = xi * xi * xi;

        double sigmaSq = sigma * sigma;

        double tMu = -xi / sigma;
        double tMuSq = tMu * tMu;
        double tMuSigma = xi / sigmaSq;
        double tMuXi = -1 / sigma;

        #pragma omp parallel for simd reduction(arma_mat_sum:hessian)
        for (unsigned int i = 0; i < y_.n_elem; ++i) {
            if (z_[i] == thisZ_) {
                double t = 1 + xi * (y_[i] - mu) / sigma;
                double logT = log(t);
                double tSq = t * t;
                double tInvXi = pow(t, -1 / xi);
                double tSigma = -(xi / sigmaSq) * (y_[i] - mu);
                double tSigmaSq = tSigma * tSigma;
                double tXi = (y_[i] - mu) / sigma;
                double tXiSq = tXi * tXi;
                double tSigmaSigma = ((2 * xi) / (sigma * sigma * sigma)) * (y_[i] - mu);
                double tSigmaXi = -(1 / sigmaSq) * (y_[i] - mu);

                // NOTE(mgnb): there are probably a lot of simplications to make here, but there's no
                // real benefit apart from brevity
                hessian(0, 0) += -(1 + 1 / xi) * (tMuSq / tSq) * (-1 + (1 / xi) * tInvXi);
                hessian(0, 1) += (
                    - (1 + 1 / xi) * (tMuSigma / t - tMu * tSigma / tSq)
                    + (1 / xi) * (tMuSigma * tInvXi / t - (1 + 1 / xi) * tMu * tSigma * tInvXi / tSq)
                );
                hessian(0, 2) += (
                    (1 / xiSq) * (tMu / t) * (1 - tInvXi)
                    - (1 + 1 / xi) * (tMuXi / t - tMu * tXi / tSq)
                    + (1 / xi) * (tInvXi / t) * (
                        tMuXi + tMu * (
                            (1 / xiSq) * logT
                            - (1 + 1 / xi) * tXi / t
                        )
                    )
                );
                hessian(1, 1) += (
                    1 / sigmaSq
                    - (1 + 1 / xi) * (tSigmaSigma / t - tSigmaSq / tSq)
                    + (1 / xi) * (tSigmaSigma * tInvXi / t - (1 + 1 / xi) * tSigmaSq * tInvXi / tSq)
                );
                hessian(1, 2) += (
                    (1 / xiSq) * (tSigma / t - tSigma * tInvXi / t)
                    - (1 + 1 / xi) * (tSigmaXi / t - tSigma * tXi / tSq)
                    + (1 / xi) * (
                        tSigmaXi * tInvXi / t
                        + tSigma * (tInvXi / t) * (
                            (1 / xiSq) * logT
                            - (1 + 1 / xi) * tXi / t
                        )
                    )
                );
                double x = (1 / xiSq) * logT - (1 / xi) * tXi / t;
                hessian(2, 2) += (
                    (-2 / xiCb) * logT
                    + (2 / xiSq) * tXi / t
                    + (1 + 1 / xi) * tXiSq / tSq
                    - tInvXi * (
                        x * x
                        - (2 / xiCb) * logT
                        + (2 / xiSq) * tXi / t
                        + (1 / xi) * tXiSq / tSq
                    )
                );
            }
        }
        break;
    }
    case GENERALISED_GAMMA: {
        double mu = parameters[0];
        double sigma = parameters[1];
        double Q = parameters[2];

        if (fabs(Q) < 0.01) {
            Q = Q < 0 ? -0.01 : 0.01;
        }

        double sumExpQW = 0;
        double sumWExpQW = 0;
        double sumWSquaredExpQW = 0;

        #pragma omp parallel for simd reduction(+:sumExpQW, sumWExpQW, sumWSquaredExpQW)
        for (unsigned int i = 0; i < y_.n_elem; ++i) {
            if (z_[i] == thisZ_) {
                double w = (logY_[i] - mu) / sigma;
                double expQw = exp(Q * w);

                sumExpQW += expQw;
                sumWExpQW += w * expQw;
                sumWSquaredExpQW += w * w * expQw;
            }
        }

        double sigmaSq = sigma * sigma;
        double sigmaCb = sigma * sigma * sigma;
        double QSq = Q * Q;
        double QCb = Q * Q * Q;

        hessian(0, 0) = -sumExpQW / sigmaSq;
        hessian(0, 1) = thisN_ / (sigmaSq * Q) - sumExpQW / (sigmaSq * Q) - sumWExpQW / sigmaSq;
        hessian(0, 2) = thisN_ / (sigma * QSq) - sumExpQW / (sigma * QSq) + sumWExpQW / (sigma * Q);
        hessian(1, 1) = (
            thisN_ / sigmaSq
            - 2 * thisN_ * mu / (sigmaCb * Q)
            + 2 * sumLogY_ / (sigmaCb * Q)
            - 2 * sumWExpQW / (sigmaSq * Q)
            - sumWSquaredExpQW / sigmaSq
        );
        hessian(1, 2) = (
            - thisN_ * mu / (sigmaSq * QSq)
            + sumLogY_ / (sigmaSq * QSq)
            - sumWExpQW / (sigma * QSq)
            + sumWSquaredExpQW / (sigma * Q)
        );
        hessian(2, 2) = (
            thisN_ * (
                - 1 / QSq
                + 6 * log(1 / QSq) / (QSq * QSq)
                + 10 / (QSq * QSq)
                - 6 * R::digamma(1 / QSq) / (QSq * QSq)
                - 4 * R::trigamma(1 / QSq) / (QCb * QCb)
                - 2 * mu / (sigma * QCb)
            )
            + 2 * sumLogY_ / (sigma * QCb)
            - 6 * sumExpQW / (QSq * QSq)
            + 4 * sumWExpQW / QCb
            - sumWSquaredExpQW / QSq
        );
    }
    default:
        break;
    }

    return symmatu(hessian);
}
