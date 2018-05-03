#include <RcppArmadillo.h>
#include "hypercube4.hpp"
#include "logistic-utils.hpp"
#include "utils.hpp"

using arma::colvec;
using arma::conv_to;
using arma::cube;
using arma::field;
using arma::mat;
using arma::rowvec;
using arma::ucolvec;

using Rcpp::as;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;
using Rcpp::stop;
using Rcpp::wrap;

// [[Rcpp::export(name=".logistic_ergodic_p")]]
NumericMatrix logisticErgodicP(
    NumericMatrix deltaSamplesR, NumericMatrix zSamplesR, IntegerVector z0SamplesR,
    NumericMatrix designMatrixR, unsigned int order
) {
    mat deltaSamples = as<mat>(deltaSamplesR);
    mat zSamples = as<mat>(zSamplesR);
    ucolvec z0Samples = as<ucolvec>(z0SamplesR);

    if (deltaSamples.n_rows != zSamples.n_rows) {
        stop("Z and delta samples must be balanced");
    }
    if (order > 1) {
        stop("Only order 1 supported");
    }

    mat designMatrix = as<mat>(designMatrixR);
    // Make room for the z's
    designMatrix.resize(designMatrix.n_rows, designMatrix.n_cols + 2 * order);

    mat pHat(designMatrix.n_rows, 2, arma::fill::zeros);
    unsigned int nDeltas = deltaSamples.n_cols / 2;

    #pragma omp parallel for
    for (unsigned int sample = 0; sample < deltaSamples.n_rows; ++sample) {
        designMatrix(0, nDeltas - 2) = z0Samples[sample] == 2;
        designMatrix(0, nDeltas - 1) = z0Samples[sample] == 3;
        for (unsigned int i = 0; i < zSamples.n_cols - 1; ++i) {
            designMatrix(i + 1, nDeltas - 2) = zSamples(sample, i) == 2;
            designMatrix(i + 1, nDeltas - 1) = zSamples(sample, i) == 3;
        }

        pHat += getLogisticP(
            conv_to<colvec>::from(deltaSamples.row(sample)),
            designMatrix
        );
    }

    pHat /= static_cast<double>(deltaSamples.n_rows);

    mat pHatOut(designMatrix.n_rows, 3);
    pHatOut.col(0) = 1 - pHat.col(0) - pHat.col(1);
    pHatOut.col(1) = pHat.col(0);
    pHatOut.col(2) = pHat.col(1);

    return wrap(pHatOut);
}

class PredictedPGenerator {
 public:
    explicit PredictedPGenerator(const mat delta, const mat designMatrix, unsigned int z0, unsigned int order)
        : delta_(delta),
          designMatrix_(designMatrix),
          order_(order),
          currentIndex_(0) {
        if (order_ > 0) {
            previousP_.set_size(delta.n_rows + 1);
            previousP_.zeros();
            previousP_[z0 - 1] = 1;
        }
    }

    colvec getNext() {
        unsigned int nComponents = delta_.n_rows + 1;

        colvec output(nComponents);

        if (order_ == 0) {
            colvec sums = delta_ * designMatrix_.row(currentIndex_).t();
            // The first value is always 0 by design
            sums.insert_rows(0, 1, true);
            output = exp(sums - sums.max());
            output /= arma::sum(output);
        } else {
            // Indices are [from, to]
            mat pTransition(nComponents, nComponents, arma::fill::zeros);
            for (unsigned int k = 0; k < nComponents; ++k) {
                if (k == 0) continue;

                double sum = arma::dot(
                    delta_.row(k - 1).head(delta_.n_cols - nComponents + 1),
                    designMatrix_.row(currentIndex_)
                );

                for (unsigned int kk = 0; kk < nComponents; ++kk) {
                    if (kk == 0) {
                        pTransition(kk, k) = sum;
                    } else {
                        pTransition(kk, k) = sum + delta_(k - 1, delta_.n_cols + kk - 1 - delta_.n_rows);
                    }
                }
            }
            pTransition = exp(pTransition - pTransition.max());
            for (unsigned int k = 0; k < nComponents; ++k) {
                pTransition.row(k) /= arma::sum(pTransition.row(k));
            }

            for (unsigned int k = 0; k < nComponents; ++k) {
                output[k] = arma::dot(previousP_, pTransition.col(k));
            }

            previousP_ = output;
        }

        currentIndex_++;

        return output;
    }

 private:
    const mat delta_;
    const mat designMatrix_;
    unsigned int order_;

    colvec previousP_;
    unsigned int currentIndex_;
};

mat logisticPredictedPLevel(
    const cube deltaSamples, const ucolvec z0Samples, const mat designMatrix, unsigned int order
) {
    mat pHat(designMatrix.n_rows, deltaSamples.n_rows + 1, arma::fill::zeros);

    for (unsigned int sampleIndex = 0; sampleIndex < deltaSamples.n_slices; ++sampleIndex) {
        PredictedPGenerator pGenerator = PredictedPGenerator(
            deltaSamples.slice(sampleIndex), designMatrix, z0Samples[sampleIndex], order
        );

        for (unsigned int i = 0; i < designMatrix.n_rows; ++i) {
            pHat.row(i) += pGenerator.getNext().t();
        }
    }

    return pHat / static_cast<double>(deltaSamples.n_slices);
}

// [[Rcpp::export(name=".logistic_predicted_p")]]
List logisticPredictedP(
    List levels, unsigned int order
) {
    field<cube> panelDelta(levels.length());
    field<ucolvec> panelZ0(levels.length());
    field<mat> panelDesignMatrix(levels.length());

    for (unsigned int levelIndex = 0; levelIndex < levels.length(); ++levelIndex) {
        List level = levels[levelIndex];
        panelDelta[levelIndex] = as<cube>(level["delta"]);
        panelZ0[levelIndex] = as<ucolvec>(level["z0"]);
        panelDesignMatrix[levelIndex] = as<mat>(level["design_matrix"]);
    }

    field<mat> results(levels.length());

    #pragma omp parallel for
    for (unsigned int levelIndex = 0; levelIndex < levels.length(); ++levelIndex) {
        results[levelIndex] = logisticPredictedPLevel(
            panelDelta[levelIndex], panelZ0[levelIndex], panelDesignMatrix[levelIndex], order
        );
    }

    return listFromField(results);
}

mat logisticMomentsLevel(
    const field<mat> distributionSamples, const cube deltaSamples, const ucolvec z0Samples,
    const mat designMatrix, unsigned int order, bool conditionOnPositive
) {
    // Output
    mat moments(designMatrix.n_rows, 3, arma::fill::zeros);
    unsigned int nComponents = distributionSamples.n_elem;

    for (unsigned int sampleIndex = 0; sampleIndex < deltaSamples.n_slices; ++sampleIndex) {
        PredictedPGenerator pGenerator = PredictedPGenerator(
            deltaSamples.slice(sampleIndex), designMatrix, z0Samples[sampleIndex], order
        );

        for (unsigned int i = 0; i < designMatrix.n_rows; ++i) {
            colvec p = pGenerator.getNext();

            if (conditionOnPositive) {
                double pPositive = arma::sum(p) - p[0];
                p /= pPositive;
            }

            colvec means(nComponents);
            colvec variances(nComponents);
            colvec skews(nComponents);

            for (unsigned int k = 0; k < nComponents; ++k) {
                double alpha = distributionSamples[k](sampleIndex, 0);
                double beta = distributionSamples[k](sampleIndex, 1);
                means[k] = p[k + 1] * alpha * beta;
                variances[k] = p[k + 1] * p[k + 1] * alpha * beta * beta;
                skews[k] = 2 / sqrt(alpha);
            }

            moments(i, 0) += arma::sum(means);
            moments(i, 1) += sqrt(arma::sum(variances));
            moments(i, 2) += (
                arma::dot(skews, variances % sqrt(variances))
                / (moments(i, 1) * moments(i, 1) * moments(i, 1))
            );
        }
    }

    return moments / static_cast<double>(deltaSamples.n_slices);
}

// [[Rcpp::export(name=".logistic_moments")]]
List logisticMoments(List distributionSamplesR, List levels, unsigned int order, bool conditionOnPositive) {
    field<mat> distributionSamples = fieldFromList<mat>(distributionSamplesR);
    field<cube> panelDelta(levels.length());
    field<ucolvec> panelZ0(levels.length());
    field<mat> panelDesignMatrix(levels.length());

    for (unsigned int levelIndex = 0; levelIndex < levels.length(); ++levelIndex) {
        List level = levels[levelIndex];
        panelDelta[levelIndex] = as<cube>(level["delta"]);
        panelZ0[levelIndex] = as<ucolvec>(level["z0"]);
        panelDesignMatrix[levelIndex] = as<mat>(level["design_matrix"]);
    }

    field<mat> results(levels.length());

    #pragma omp parallel for
    for (unsigned int levelIndex = 0; levelIndex < levels.length(); ++levelIndex) {
        results[levelIndex] = logisticMomentsLevel(
            distributionSamples,
            panelDelta[levelIndex], panelZ0[levelIndex], panelDesignMatrix[levelIndex],
            order, conditionOnPositive
        );
    }

    return listFromField(results);
}

// [[Rcpp::export(name=".logistic_fitted_delta")]]
NumericVector logisticFittedDelta(
    NumericVector deltaFamilyMeanSamples,
    NumericMatrix levelDesignMatrixR,
    NumericVector probsR
) {
    mat levelDesignMatrix = arma::trans(as<mat>(levelDesignMatrixR));
    colvec probs = as<colvec>(probsR);

    NumericVector dims = deltaFamilyMeanSamples.attr("dim");

    unsigned int nLevels = levelDesignMatrix.n_cols;
    unsigned int nSamples = dims[0];
    unsigned int nVars = dims[1];
    unsigned int nColumns = dims[2];
    unsigned int nRows = dims[3];

    unsigned int nOutputs = probs.n_elem + 1;

    ucolvec probIndices(probs.n_elem);
    colvec probIndicesAdjustment(probs.n_elem);
    for (unsigned int i = 0; i < probs.n_elem; ++i) {
        double index = static_cast<double>(nSamples - 1) * probs[i];
        probIndices[i] = floor(index);
        probIndicesAdjustment[i] = index - floor(index);
    }

    hypercube4 output;
    output.set_size(nLevels, nColumns, nOutputs, nRows);
    cube deltaFamilySubset(nSamples, nVars, nColumns);
    cube outputTemp(nLevels, nColumns, nOutputs);

    for (unsigned int row = 0; row < nRows; ++row) {
        std::copy(
            deltaFamilyMeanSamples.begin() + row * nSamples * nVars * nColumns,
            deltaFamilyMeanSamples.begin() + (row + 1) * nSamples * nVars * nColumns,
            deltaFamilySubset.begin()
        );

        for (unsigned int column = 0; column < nColumns; ++column) {
            // Sort each column
            mat temp = arma::sort(
                deltaFamilySubset.slice(column) * levelDesignMatrix,
                "ascend", 0
            );
            outputTemp.slice(0).col(column) = arma::mean(temp, 0).t();
            for (unsigned int i = 0; i < probs.n_elem; ++i) {
                if (probIndices[i] == nSamples - 1) {
                    // Boundary condition hit
                    outputTemp.slice(i + 1).col(column) = temp.row(probIndices[i]).t();
                } else {
                    // Average between index and next up
                    outputTemp.slice(i + 1).col(column) = (
                        (1 - probIndicesAdjustment[i]) * temp.row(probIndices[i])
                        + probIndicesAdjustment[i] * temp.row(probIndices[i] + 1)
                    ).t();
                }
            }
        }

        output.set_hyperslice(row, outputTemp);
    }

    return output.asNumericVector();
}
