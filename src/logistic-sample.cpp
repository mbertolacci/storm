#include <chrono>
#include <RcppArmadillo.h>
#include <vector>

#include "utils.hpp"
#include "distribution.hpp"
#include "parameter-sampler.hpp"
#include "logistic-parameter-sampler.hpp"
#include "logistic-utils.hpp"

using arma::colvec;
using arma::cube;
using arma::field;
using arma::mat;
using arma::ucolvec;
using arma::umat;

using Rcpp::as;
using Rcpp::checkUserInterrupt;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;
using Rcpp::Rcout;
using Rcpp::StringVector;
using Rcpp::wrap;

mat sampleDeltaFamilyMean(cube panelDeltaCurrent, mat deltaFamilyVarianceCurrent, mat parameters) {
    double nLevels = panelDeltaCurrent.n_slices;

    mat deltaSum = sum(panelDeltaCurrent, 2);
    mat output(size(deltaSum));

    for (unsigned int i = 0; i < output.n_elem; ++i) {
        double variance = 1 / (
            nLevels / deltaFamilyVarianceCurrent[i]
            + 1 / parameters(i, 1)
        );
        output[i] = R::rnorm(
            variance * (deltaSum[i] / deltaFamilyVarianceCurrent[i] + parameters(i, 0) / parameters(i, 1)),
            variance
        );
    }

    return output;
}

mat sampleDeltaFamilyVariance(cube panelDeltaCurrent, mat deltaFamilyMeanCurrent, mat parameters) {
    double nLevels = panelDeltaCurrent.n_slices;

    mat deltaSumDeviation = sum(square(panelDeltaCurrent.each_slice() - deltaFamilyMeanCurrent), 2);

    mat output(size(deltaSumDeviation));
    for (unsigned int i = 0; i < output.n_elem; ++i) {
        output[i] = 1 / R::rgamma(
            parameters(i, 2) + nLevels / 2,
            1 / (parameters(i, 3) + 0.5 * deltaSumDeviation[i])
        );
    }

    return output;
}

// [[Rcpp::export(name=".ptsm_logistic_sample")]]
List logisticSample(
    int nSamples, int burnIn,
    List panelYR, List panelExplanatoryVariablesR, unsigned int order,
    StringVector distributionNames, List priors, List samplingSchemes,
    List panelZStart, List thetaStart,
    List panelDeltaStart, NumericMatrix deltaFamilyMeanStart, NumericMatrix deltaFamilyVarianceStart,
    int thetaSampleThinning = 1, int zSampleThinning = 0, int yMissingSampleThinning = 0,
    unsigned int verbose = 0
) {
    int nIterations = nSamples + burnIn;

    unsigned int nLevels = panelExplanatoryVariablesR.length();

    field<mat> panelExplanatoryVariables = fieldFromList<mat>(panelExplanatoryVariablesR);
    if (order > 0) {
        for (unsigned int level = 0; level < nLevels; ++level) {
            // Add room for the latent variable lags
            panelExplanatoryVariables[level].resize(
                panelExplanatoryVariables[level].n_rows,
                panelExplanatoryVariables[level].n_cols + 2 * order
            );
        }
    }
    unsigned int nDeltas = panelExplanatoryVariables[0].n_cols;

    // Priors and samplers
    Distribution lowerDistribution(distributionNames[0]);
    Distribution upperDistribution(distributionNames[1]);

    ParameterSampler lowerSampler(priors[0], samplingSchemes[0], lowerDistribution);
    ParameterSampler upperSampler(priors[1], samplingSchemes[1], upperDistribution);

    List logisticPriorConfiguration = priors["logistic"];
    mat logisticPriorParameters = as<mat>(logisticPriorConfiguration["parameters"]);
    LogisticParameterPrior logisticParameterPrior(logisticPriorConfiguration);
    std::vector<LogisticParameterSampler> panelLogisticParameterSampler;
    for (unsigned int level = 0; level < nLevels; ++level) {
        panelLogisticParameterSampler.push_back(
            LogisticParameterSampler(as<List>(as<List>(samplingSchemes["logistic"])[level]))
        );
    }
    // Data and missing data
    unsigned int panelTotalN = 0;
    field<colvec> panelY = fieldFromList<colvec>(panelYR);
    field<colvec> panelLogY(nLevels);
    for (unsigned int level = 0; level < nLevels; ++level) {
        panelTotalN += panelY[level].n_elem;
        panelLogY[level] = log(panelY[level]);
    }
    field<ucolvec> panelYMissingIndices(nLevels);
    std::vector< std::vector<bool> > panelYIsMissing(nLevels);
    for (unsigned int level = 0; level < nLevels; ++level) {
        MissingValuesPair missingValuesPair = findMissingValues(panelYR[level]);
        panelYMissingIndices[level] = missingValuesPair.first;
        panelYIsMissing[level] = missingValuesPair.second;
    }

    // Starting values for parameters
    ucolvec panelZ0Current(nLevels, arma::fill::zeros);
    field<ucolvec> panelZCurrent = fieldFromList<ucolvec>(panelZStart);
    colvec thetaLower = as<colvec>(thetaStart[0]);
    colvec thetaUpper = as<colvec>(thetaStart[1]);
    cube panelDeltaCurrent = cubeFromList(panelDeltaStart);

    mat deltaFamilyMeanCurrent = as<mat>(deltaFamilyMeanStart);
    mat deltaFamilyVarianceCurrent = as<mat>(deltaFamilyVarianceStart);

    // Computed from parameters
    field<mat> panelPCurrent(nLevels);

    // Samples for output
    mat thetaSample;
    if (thetaSampleThinning > 0) {
        if (strcmp(logisticPriorConfiguration["type"], "hierarchical") == 0) {
            thetaSample = mat(
                ceil(static_cast<double>(nSamples) / static_cast<double>(thetaSampleThinning)),
                thetaLower.n_elem + thetaUpper.n_elem + 2 * (nLevels + 2) * nDeltas
            );
        } else {
            thetaSample = mat(
                ceil(static_cast<double>(nSamples) / static_cast<double>(thetaSampleThinning)),
                thetaLower.n_elem + thetaUpper.n_elem + 2 * nLevels * nDeltas
            );
        }
    }
    umat panelZ0Sample;
    field<umat> panelZSample;
    if (zSampleThinning > 0) {
        int nZSamples = ceil(static_cast<double>(nSamples) / static_cast<double>(zSampleThinning));
        panelZ0Sample = umat(nZSamples, nLevels);
        panelZSample = field<umat>(nLevels);
        for (unsigned int level = 0; level < nLevels; ++level) {
            panelZSample[level] = umat(nZSamples, panelY[level].n_elem);
        }
    }
    field<mat> panelYMissingSample;
    if (yMissingSampleThinning > 0) {
        int nYMissingSamples = ceil(static_cast<double>(nSamples) / static_cast<double>(yMissingSampleThinning));
        panelYMissingSample = field<mat>(nLevels);
        for (unsigned int level = 0; level < nLevels; ++level) {
            panelYMissingSample[level] = mat(nYMissingSamples, panelYMissingIndices[level].size());
        }
    }

    // Initial missing value sample
    for (unsigned int level = 0; level < nLevels; ++level) {
        sampleMissingY(
            panelY[level], panelLogY[level], panelYMissingIndices[level], panelZCurrent[level],
            ParameterBoundDistribution(thetaLower, lowerDistribution),
            ParameterBoundDistribution(thetaUpper, upperDistribution)
        );

        if (order > 0) {
            panelExplanatoryVariables[level](0, nDeltas - 2) = panelZ0Current[level] == 2;
            panelExplanatoryVariables[level](0, nDeltas - 1) = panelZ0Current[level] == 3;
            for (unsigned int i = 0; i < panelZCurrent[level].n_elem - 1; ++i) {
                panelExplanatoryVariables[level](i + 1, nDeltas - 2) = panelZCurrent[level][i] == 2;
                panelExplanatoryVariables[level](i + 1, nDeltas - 1) = panelZCurrent[level][i] == 3;
            }
        }
    }

    // Set initial pCurrent
    for (unsigned int level = 0; level < nLevels; ++level) {
        panelPCurrent[level] = getLogisticP(panelDeltaCurrent.slice(level), panelExplanatoryVariables[level]);
    }

    std::chrono::time_point<std::chrono::system_clock> startIteration, endIteration;
    startIteration = std::chrono::system_clock::now();

    for (int iteration = 0; iteration < nIterations; ++iteration) {
        if (strcmp(logisticPriorConfiguration["type"], "hierarchical") == 0) {
            // Sample \bm{\sigma}^2
            deltaFamilyVarianceCurrent = sampleDeltaFamilyVariance(
                panelDeltaCurrent, deltaFamilyMeanCurrent, logisticPriorParameters
            );

            // Sample /bm{\mu}
            deltaFamilyMeanCurrent = sampleDeltaFamilyMean(
                panelDeltaCurrent, deltaFamilyVarianceCurrent, logisticPriorParameters
            );
        }

        colvec y = vectorise(panelY);
        colvec logY = vectorise(panelLogY);
        ucolvec zCurrent = vectorise(panelZCurrent);
        // Sample mixture 2
        thetaLower = lowerSampler.sample(thetaLower, DataBoundDistribution(y, logY, zCurrent, 2, lowerDistribution));
        // Sample mixture 3
        thetaUpper = upperSampler.sample(thetaUpper, DataBoundDistribution(y, logY, zCurrent, 3, upperDistribution));

        LogisticParameterPrior boundLogisticParameterPrior = logisticParameterPrior.withParameters(
            deltaFamilyMeanCurrent,
            deltaFamilyVarianceCurrent
        );

        #pragma omp parallel for
        for (unsigned int level = 0; level < nLevels; ++level) {
            // Sample \bm{z}
            panelZ0Current[level] = sampleSingleZ(panelPCurrent[level](0, 0), panelPCurrent[level](0, 1));
            panelZCurrent[level] = sampleZ(
                panelPCurrent[level], panelY[level], panelYIsMissing[level],
                ParameterBoundDistribution(thetaLower, lowerDistribution),
                ParameterBoundDistribution(thetaUpper, upperDistribution)
            );

            // Sample missing values \bm{y^*}
            sampleMissingY(
                panelY[level], panelLogY[level], panelYMissingIndices[level], panelZCurrent[level],
                ParameterBoundDistribution(thetaLower, lowerDistribution),
                ParameterBoundDistribution(thetaUpper, upperDistribution)
            );

            if (order > 0) {
                panelExplanatoryVariables[level](0, nDeltas - 2) = panelZ0Current[level] == 2;
                panelExplanatoryVariables[level](0, nDeltas - 1) = panelZ0Current[level] == 3;
                for (unsigned int i = 0; i < panelZCurrent[level].n_elem - 1; ++i) {
                    panelExplanatoryVariables[level](i + 1, nDeltas - 2) = panelZCurrent[level][i] == 2;
                    panelExplanatoryVariables[level](i + 1, nDeltas - 1) = panelZCurrent[level][i] == 3;
                }
            }

            // Sample \Delta_s
            panelDeltaCurrent.slice(level) = panelLogisticParameterSampler[level].sample(
                panelDeltaCurrent.slice(level),
                panelPCurrent[level],
                panelZCurrent[level],
                panelExplanatoryVariables[level],
                boundLogisticParameterPrior
            );
            panelPCurrent[level] = getLogisticP(panelDeltaCurrent.slice(level), panelExplanatoryVariables[level]);
        }

        if (verbose > 0 && iteration % verbose == 0) {
            Rcout << "in " << iteration << " have "
                << (NumericVector)wrap(thetaLower) << " | "
                << (NumericVector)wrap(thetaUpper) << " | \n";

            for (unsigned int level = 0; level < nLevels; ++level) {
                Rcout << "z0 = " << panelZ0Current[level] << "\n";
                Rcout << "delta" << level << " =\n" << panelDeltaCurrent.slice(level);
            }
            Rcout << "family means =\n" << deltaFamilyMeanCurrent;
            Rcout << "family variances =\n" << deltaFamilyVarianceCurrent;

            endIteration = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = endIteration - startIteration;
            double timePerIteration = (1000 * elapsed_seconds.count() / verbose);
            Rcout << "time per iteration is " << timePerIteration << "ms ";
            Rcout << "(estimated time remaining " << ((nIterations - iteration) * timePerIteration) / 1000 << "s)\n";
            startIteration = endIteration;

            // NOTE(mgnb): checks whether the user has pressed Ctrl-C (among other things)
            checkUserInterrupt();
        }

        if (verbose > 0 && iteration == burnIn) {
            Rcout << "Burn in complete\n";
            lowerSampler.printAcceptanceRatio(burnIn);
            upperSampler.printAcceptanceRatio(burnIn);
            lowerSampler.resetAcceptCount();
            upperSampler.resetAcceptCount();

            for (unsigned int level = 0; level < nLevels; ++level) {
                panelLogisticParameterSampler[level].printAcceptanceRatios(burnIn);
                panelLogisticParameterSampler[level].resetAcceptCounts();
            }
        }

        if (iteration >= burnIn) {
            int index = iteration - burnIn;

            if (thetaSampleThinning > 0 && (index % thetaSampleThinning == 0)) {
                if (strcmp(logisticPriorConfiguration["type"], "hierarchical") == 0) {
                    copyMultiple(
                        thetaSample.begin_row(index / thetaSampleThinning),
                        thetaLower, thetaUpper,
                        vectorise(panelDeltaCurrent, 2, 0, 1),
                        deltaFamilyMeanCurrent.t().eval(),
                        deltaFamilyVarianceCurrent.t().eval()
                    );
                } else {
                    copyMultiple(
                        thetaSample.begin_row(index / thetaSampleThinning),
                        thetaLower, thetaUpper,
                        vectorise(panelDeltaCurrent, 2, 0, 1)
                    );
                }
            }

            if (zSampleThinning > 0 && (index % zSampleThinning == 0)) {
                unsigned int zSampleIndex = index / zSampleThinning;

                for (unsigned int level = 0; level < nLevels; ++level) {
                    panelZ0Sample(zSampleIndex, level) = panelZ0Current[level];
                    panelZSample[level].row(zSampleIndex) = panelZCurrent[level].t();
                }
            }

            if (yMissingSampleThinning > 0 && (index % yMissingSampleThinning == 0)) {
                int yMissingSampleIndex = index / yMissingSampleThinning;

                for (unsigned int level = 0; level < nLevels; ++level) {
                    panelYMissingSample[level].row(yMissingSampleIndex)
                        = panelY[level](panelYMissingIndices[level]).t();
                }
            }
        }
    }

    if (verbose > 0) {
        lowerSampler.printAcceptanceRatio(nSamples);
        upperSampler.printAcceptanceRatio(nSamples);

        for (unsigned int level = 0; level < nLevels; ++level) {
            panelLogisticParameterSampler[level].printAcceptanceRatios(nSamples);
        }
    }

    List results;
    if (thetaSampleThinning > 0) {
        results["theta_sample"] = wrap(thetaSample);
    }
    if (zSampleThinning > 0) {
        results["panel_z0_sample"] = wrap(panelZ0Sample);
        results["panel_z_sample"] = listFromField(panelZSample);
    }
    if (yMissingSampleThinning > 0) {
        results["panel_y_missing_sample"] = listFromField(panelYMissingSample);
    }

    return results;
}

// [[Rcpp::export(name=".ptsm_logistic_sample_y")]]
List logisticSampleY(
    List panelExplanatoryVariablesR, List panelDeltaSampleR,
    NumericMatrix thetaLowerSampleR, NumericMatrix thetaUpperSampleR,
    StringVector distributionNames
) {
    mat thetaLowerSample = trans(as<mat>(thetaLowerSampleR));
    mat thetaUpperSample = trans(as<mat>(thetaUpperSampleR));
    Distribution lowerDistribution(distributionNames[0]);
    Distribution upperDistribution(distributionNames[1]);

    unsigned int nLevels = panelExplanatoryVariablesR.length();

    field<mat> panelExplanatoryVariables = fieldFromList<mat>(panelExplanatoryVariablesR);
    for (unsigned int level = 0; level < nLevels; ++level) {
        // Add room for the latent variable lags
        panelExplanatoryVariables[level].resize(
            panelExplanatoryVariables[level].n_rows,
            panelExplanatoryVariables[level].n_cols + 2
        );
    }

    NumericMatrix base = panelDeltaSampleR[0];
    cube panelDeltaSample(base.ncol(), base.nrow(), panelDeltaSampleR.length());
    for (unsigned int level = 0; level < panelDeltaSampleR.length(); ++level) {
        panelDeltaSample.slice(level) = trans(as<mat>(panelDeltaSampleR[level]));
    }

    unsigned int nSamples = panelDeltaSample.n_cols;
    unsigned int nDeltas = panelExplanatoryVariables[0].n_cols;

    // Output
    field<mat> panelYSample(nLevels);
    field<umat> panelYSampleZ(nLevels);
    for (unsigned int level = 0; level < nLevels; ++level) {
        panelYSample[level] = mat(nSamples, panelExplanatoryVariables[level].n_rows);
        panelYSampleZ[level] = umat(nSamples, panelExplanatoryVariables[level].n_rows);
    }

    for (unsigned int sampleIndex = 0; sampleIndex < nSamples; ++sampleIndex) {
        #pragma omp parallel for
        for (unsigned int level = 0; level < nLevels; ++level) {
            colvec delta = panelDeltaSample.slice(level).col(sampleIndex);
            unsigned int previousZ = 1;

            for (unsigned int i = 0; i < panelYSample[level].n_cols; ++i) {
                panelExplanatoryVariables[level](i, nDeltas - 2) = previousZ == 2;
                panelExplanatoryVariables[level](i, nDeltas - 1) = previousZ == 3;
                double lowerSum = dot(delta.head(nDeltas), panelExplanatoryVariables[level].row(i));
                double upperSum = dot(delta.tail(nDeltas), panelExplanatoryVariables[level].row(i));

                double expDiff = exp(upperSum - lowerSum);
                double pLower = 1 / (1 + exp(-lowerSum) + expDiff);
                double pUpper = 1 / (1 + exp(-upperSum) + 1 / expDiff);

                unsigned int z = sampleSingleZ(pLower, pUpper);
                double y = 0;

                if (z == 2) {
                    y = lowerDistribution.sample(thetaLowerSample.col(sampleIndex));
                } else if (z == 3) {
                    y = upperDistribution.sample(thetaUpperSample.col(sampleIndex));
                }

                panelYSample[level](sampleIndex, i) = y;
                panelYSampleZ[level](sampleIndex, i) = z;

                previousZ = z;
            }
        }
    }

    List results;
    results["panel_y_sample"] = listFromField(panelYSample);
    results["panel_y_sample_z"] = listFromField(panelYSampleZ);

    return results;
}
