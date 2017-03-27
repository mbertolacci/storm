// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// idigamma
double idigamma(double y);
RcppExport SEXP positivemixtures_idigamma(SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(idigamma(y));
    return rcpp_result_gen;
END_RCPP
}
// logDensity
double logDensity(double alpha, double logBeta, double logP, double q, double r);
RcppExport SEXP positivemixtures_logDensity(SEXP alphaSEXP, SEXP logBetaSEXP, SEXP logPSEXP, SEXP qSEXP, SEXP rSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type logBeta(logBetaSEXP);
    Rcpp::traits::input_parameter< double >::type logP(logPSEXP);
    Rcpp::traits::input_parameter< double >::type q(qSEXP);
    Rcpp::traits::input_parameter< double >::type r(rSEXP);
    rcpp_result_gen = Rcpp::wrap(logDensity(alpha, logBeta, logP, q, r));
    return rcpp_result_gen;
END_RCPP
}
// rGammaShapeConjugateR
NumericVector rGammaShapeConjugateR(unsigned int n, double beta, double logP, double q, double r, Rcpp::Nullable<NumericVector> x, Rcpp::Nullable<NumericVector> prior);
RcppExport SEXP positivemixtures_rGammaShapeConjugateR(SEXP nSEXP, SEXP betaSEXP, SEXP logPSEXP, SEXP qSEXP, SEXP rSEXP, SEXP xSEXP, SEXP priorSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type n(nSEXP);
    Rcpp::traits::input_parameter< double >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type logP(logPSEXP);
    Rcpp::traits::input_parameter< double >::type q(qSEXP);
    Rcpp::traits::input_parameter< double >::type r(rSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<NumericVector> >::type x(xSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<NumericVector> >::type prior(priorSEXP);
    rcpp_result_gen = Rcpp::wrap(rGammaShapeConjugateR(n, beta, logP, q, r, x, prior));
    return rcpp_result_gen;
END_RCPP
}
// gpSample
List gpSample(NumericVector y, NumericMatrix designMatrix, NumericVector betaPriorMean, NumericVector betaPriorVariance, double variancePriorAlpha, double variancePriorBeta, double tauSquaredPriorAlpha, double tauSquaredPriorBeta, unsigned int nGPBases);
RcppExport SEXP positivemixtures_gpSample(SEXP ySEXP, SEXP designMatrixSEXP, SEXP betaPriorMeanSEXP, SEXP betaPriorVarianceSEXP, SEXP variancePriorAlphaSEXP, SEXP variancePriorBetaSEXP, SEXP tauSquaredPriorAlphaSEXP, SEXP tauSquaredPriorBetaSEXP, SEXP nGPBasesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type designMatrix(designMatrixSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type betaPriorMean(betaPriorMeanSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type betaPriorVariance(betaPriorVarianceSEXP);
    Rcpp::traits::input_parameter< double >::type variancePriorAlpha(variancePriorAlphaSEXP);
    Rcpp::traits::input_parameter< double >::type variancePriorBeta(variancePriorBetaSEXP);
    Rcpp::traits::input_parameter< double >::type tauSquaredPriorAlpha(tauSquaredPriorAlphaSEXP);
    Rcpp::traits::input_parameter< double >::type tauSquaredPriorBeta(tauSquaredPriorBetaSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nGPBases(nGPBasesSEXP);
    rcpp_result_gen = Rcpp::wrap(gpSample(y, designMatrix, betaPriorMean, betaPriorVariance, variancePriorAlpha, variancePriorBeta, tauSquaredPriorAlpha, tauSquaredPriorBeta, nGPBases));
    return rcpp_result_gen;
END_RCPP
}
// rgevVector
NumericVector rgevVector(int n, double mu, double sigma, double xi);
RcppExport SEXP positivemixtures_rgevVector(SEXP nSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP xiSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< double >::type mu(muSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< double >::type xi(xiSEXP);
    rcpp_result_gen = Rcpp::wrap(rgevVector(n, mu, sigma, xi));
    return rcpp_result_gen;
END_RCPP
}
// estimatePwm
double estimatePwm(NumericVector x, int r);
RcppExport SEXP positivemixtures_estimatePwm(SEXP xSEXP, SEXP rSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type r(rSEXP);
    rcpp_result_gen = Rcpp::wrap(estimatePwm(x, r));
    return rcpp_result_gen;
END_RCPP
}
// gevPwmEstimate
NumericVector gevPwmEstimate(NumericVector x);
RcppExport SEXP positivemixtures_gevPwmEstimate(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(gevPwmEstimate(x));
    return rcpp_result_gen;
END_RCPP
}
// gevPwmEstimateConstrained
NumericVector gevPwmEstimateConstrained(NumericVector x, double supportLim);
RcppExport SEXP positivemixtures_gevPwmEstimateConstrained(SEXP xSEXP, SEXP supportLimSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< double >::type supportLim(supportLimSEXP);
    rcpp_result_gen = Rcpp::wrap(gevPwmEstimateConstrained(x, supportLim));
    return rcpp_result_gen;
END_RCPP
}
// hmmSample
List hmmSample(unsigned int nSamples, unsigned int burnIn, NumericVector yR, StringVector distributionNames, List priors, List samplingSchemes, IntegerVector zStart, List thetaStart, int thetaSampleThinning, int zSampleThinning, int yMissingSampleThinning, unsigned int verbose);
RcppExport SEXP positivemixtures_hmmSample(SEXP nSamplesSEXP, SEXP burnInSEXP, SEXP yRSEXP, SEXP distributionNamesSEXP, SEXP priorsSEXP, SEXP samplingSchemesSEXP, SEXP zStartSEXP, SEXP thetaStartSEXP, SEXP thetaSampleThinningSEXP, SEXP zSampleThinningSEXP, SEXP yMissingSampleThinningSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type nSamples(nSamplesSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type burnIn(burnInSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type yR(yRSEXP);
    Rcpp::traits::input_parameter< StringVector >::type distributionNames(distributionNamesSEXP);
    Rcpp::traits::input_parameter< List >::type priors(priorsSEXP);
    Rcpp::traits::input_parameter< List >::type samplingSchemes(samplingSchemesSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type zStart(zStartSEXP);
    Rcpp::traits::input_parameter< List >::type thetaStart(thetaStartSEXP);
    Rcpp::traits::input_parameter< int >::type thetaSampleThinning(thetaSampleThinningSEXP);
    Rcpp::traits::input_parameter< int >::type zSampleThinning(zSampleThinningSEXP);
    Rcpp::traits::input_parameter< int >::type yMissingSampleThinning(yMissingSampleThinningSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(hmmSample(nSamples, burnIn, yR, distributionNames, priors, samplingSchemes, zStart, thetaStart, thetaSampleThinning, zSampleThinning, yMissingSampleThinning, verbose));
    return rcpp_result_gen;
END_RCPP
}
// independentSample
List independentSample(unsigned int nSamples, unsigned int burnIn, NumericVector yR, StringVector distributionNames, List priors, List samplingSchemes, IntegerVector zStart, List distributionsStart, unsigned int distributionSampleThinning, unsigned int pSampleThinning, unsigned int zSampleThinning, unsigned int yMissingSampleThinning, bool progress);
RcppExport SEXP positivemixtures_independentSample(SEXP nSamplesSEXP, SEXP burnInSEXP, SEXP yRSEXP, SEXP distributionNamesSEXP, SEXP priorsSEXP, SEXP samplingSchemesSEXP, SEXP zStartSEXP, SEXP distributionsStartSEXP, SEXP distributionSampleThinningSEXP, SEXP pSampleThinningSEXP, SEXP zSampleThinningSEXP, SEXP yMissingSampleThinningSEXP, SEXP progressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type nSamples(nSamplesSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type burnIn(burnInSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type yR(yRSEXP);
    Rcpp::traits::input_parameter< StringVector >::type distributionNames(distributionNamesSEXP);
    Rcpp::traits::input_parameter< List >::type priors(priorsSEXP);
    Rcpp::traits::input_parameter< List >::type samplingSchemes(samplingSchemesSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type zStart(zStartSEXP);
    Rcpp::traits::input_parameter< List >::type distributionsStart(distributionsStartSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type distributionSampleThinning(distributionSampleThinningSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type pSampleThinning(pSampleThinningSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type zSampleThinning(zSampleThinningSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type yMissingSampleThinning(yMissingSampleThinningSEXP);
    Rcpp::traits::input_parameter< bool >::type progress(progressSEXP);
    rcpp_result_gen = Rcpp::wrap(independentSample(nSamples, burnIn, yR, distributionNames, priors, samplingSchemes, zStart, distributionsStart, distributionSampleThinning, pSampleThinning, zSampleThinning, yMissingSampleThinning, progress));
    return rcpp_result_gen;
END_RCPP
}
// logisticErgodicP
NumericMatrix logisticErgodicP(NumericMatrix deltaSamplesR, NumericMatrix zSamplesR, IntegerVector z0SamplesR, NumericMatrix designMatrixR, unsigned int order);
RcppExport SEXP positivemixtures_logisticErgodicP(SEXP deltaSamplesRSEXP, SEXP zSamplesRSEXP, SEXP z0SamplesRSEXP, SEXP designMatrixRSEXP, SEXP orderSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type deltaSamplesR(deltaSamplesRSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type zSamplesR(zSamplesRSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type z0SamplesR(z0SamplesRSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type designMatrixR(designMatrixRSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type order(orderSEXP);
    rcpp_result_gen = Rcpp::wrap(logisticErgodicP(deltaSamplesR, zSamplesR, z0SamplesR, designMatrixR, order));
    return rcpp_result_gen;
END_RCPP
}
// logisticPredictedP
List logisticPredictedP(List levels, unsigned int order);
RcppExport SEXP positivemixtures_logisticPredictedP(SEXP levelsSEXP, SEXP orderSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type levels(levelsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type order(orderSEXP);
    rcpp_result_gen = Rcpp::wrap(logisticPredictedP(levels, order));
    return rcpp_result_gen;
END_RCPP
}
// logisticMoments
List logisticMoments(List distributionSamplesR, List levels, unsigned int order, bool conditionOnPositive);
RcppExport SEXP positivemixtures_logisticMoments(SEXP distributionSamplesRSEXP, SEXP levelsSEXP, SEXP orderSEXP, SEXP conditionOnPositiveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type distributionSamplesR(distributionSamplesRSEXP);
    Rcpp::traits::input_parameter< List >::type levels(levelsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type order(orderSEXP);
    Rcpp::traits::input_parameter< bool >::type conditionOnPositive(conditionOnPositiveSEXP);
    rcpp_result_gen = Rcpp::wrap(logisticMoments(distributionSamplesR, levels, order, conditionOnPositive));
    return rcpp_result_gen;
END_RCPP
}
// logisticFittedDelta
NumericVector logisticFittedDelta(NumericVector deltaFamilyMeanSamples, NumericMatrix levelDesignMatrixR, NumericVector probsR);
RcppExport SEXP positivemixtures_logisticFittedDelta(SEXP deltaFamilyMeanSamplesSEXP, SEXP levelDesignMatrixRSEXP, SEXP probsRSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type deltaFamilyMeanSamples(deltaFamilyMeanSamplesSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type levelDesignMatrixR(levelDesignMatrixRSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type probsR(probsRSEXP);
    rcpp_result_gen = Rcpp::wrap(logisticFittedDelta(deltaFamilyMeanSamples, levelDesignMatrixR, probsR));
    return rcpp_result_gen;
END_RCPP
}
// logisticGenerate
List logisticGenerate(NumericMatrix deltaR, NumericMatrix explanatoryVariablesR, List distributionParameters, StringVector distributionNames, unsigned int order);
RcppExport SEXP positivemixtures_logisticGenerate(SEXP deltaRSEXP, SEXP explanatoryVariablesRSEXP, SEXP distributionParametersSEXP, SEXP distributionNamesSEXP, SEXP orderSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type deltaR(deltaRSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type explanatoryVariablesR(explanatoryVariablesRSEXP);
    Rcpp::traits::input_parameter< List >::type distributionParameters(distributionParametersSEXP);
    Rcpp::traits::input_parameter< StringVector >::type distributionNames(distributionNamesSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type order(orderSEXP);
    rcpp_result_gen = Rcpp::wrap(logisticGenerate(deltaR, explanatoryVariablesR, distributionParameters, distributionNames, order));
    return rcpp_result_gen;
END_RCPP
}
// logisticSampleMPI
List logisticSampleMPI(unsigned int nSamples, unsigned int burnIn, List panelY, List panelDesignMatrix, unsigned int order, StringVector distributionNames, List priors, List samplingSchemes, List thetaStart, List panelDeltaStart, NumericVector deltaFamilyMeanStart, NumericMatrix deltaFamilyVarianceStart, Rcpp::Nullable<NumericMatrix> deltaDesignMatrix, List thinning, unsigned int verbose, bool progress);
RcppExport SEXP positivemixtures_logisticSampleMPI(SEXP nSamplesSEXP, SEXP burnInSEXP, SEXP panelYSEXP, SEXP panelDesignMatrixSEXP, SEXP orderSEXP, SEXP distributionNamesSEXP, SEXP priorsSEXP, SEXP samplingSchemesSEXP, SEXP thetaStartSEXP, SEXP panelDeltaStartSEXP, SEXP deltaFamilyMeanStartSEXP, SEXP deltaFamilyVarianceStartSEXP, SEXP deltaDesignMatrixSEXP, SEXP thinningSEXP, SEXP verboseSEXP, SEXP progressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type nSamples(nSamplesSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type burnIn(burnInSEXP);
    Rcpp::traits::input_parameter< List >::type panelY(panelYSEXP);
    Rcpp::traits::input_parameter< List >::type panelDesignMatrix(panelDesignMatrixSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type order(orderSEXP);
    Rcpp::traits::input_parameter< StringVector >::type distributionNames(distributionNamesSEXP);
    Rcpp::traits::input_parameter< List >::type priors(priorsSEXP);
    Rcpp::traits::input_parameter< List >::type samplingSchemes(samplingSchemesSEXP);
    Rcpp::traits::input_parameter< List >::type thetaStart(thetaStartSEXP);
    Rcpp::traits::input_parameter< List >::type panelDeltaStart(panelDeltaStartSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type deltaFamilyMeanStart(deltaFamilyMeanStartSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type deltaFamilyVarianceStart(deltaFamilyVarianceStartSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<NumericMatrix> >::type deltaDesignMatrix(deltaDesignMatrixSEXP);
    Rcpp::traits::input_parameter< List >::type thinning(thinningSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< bool >::type progress(progressSEXP);
    rcpp_result_gen = Rcpp::wrap(logisticSampleMPI(nSamples, burnIn, panelY, panelDesignMatrix, order, distributionNames, priors, samplingSchemes, thetaStart, panelDeltaStart, deltaFamilyMeanStart, deltaFamilyVarianceStart, deltaDesignMatrix, thinning, verbose, progress));
    return rcpp_result_gen;
END_RCPP
}
// logisticSample
List logisticSample(unsigned int nSamples, unsigned int burnIn, List panelY, List panelDesignMatrix, unsigned int order, StringVector distributionNames, List priors, List samplingSchemes, List thetaStart, List panelDeltaStart, NumericVector deltaFamilyMeanStart, NumericMatrix deltaFamilyVarianceStart, Rcpp::Nullable<NumericMatrix> deltaDesignMatrix, List thinning, unsigned int verbose, bool progress);
RcppExport SEXP positivemixtures_logisticSample(SEXP nSamplesSEXP, SEXP burnInSEXP, SEXP panelYSEXP, SEXP panelDesignMatrixSEXP, SEXP orderSEXP, SEXP distributionNamesSEXP, SEXP priorsSEXP, SEXP samplingSchemesSEXP, SEXP thetaStartSEXP, SEXP panelDeltaStartSEXP, SEXP deltaFamilyMeanStartSEXP, SEXP deltaFamilyVarianceStartSEXP, SEXP deltaDesignMatrixSEXP, SEXP thinningSEXP, SEXP verboseSEXP, SEXP progressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type nSamples(nSamplesSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type burnIn(burnInSEXP);
    Rcpp::traits::input_parameter< List >::type panelY(panelYSEXP);
    Rcpp::traits::input_parameter< List >::type panelDesignMatrix(panelDesignMatrixSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type order(orderSEXP);
    Rcpp::traits::input_parameter< StringVector >::type distributionNames(distributionNamesSEXP);
    Rcpp::traits::input_parameter< List >::type priors(priorsSEXP);
    Rcpp::traits::input_parameter< List >::type samplingSchemes(samplingSchemesSEXP);
    Rcpp::traits::input_parameter< List >::type thetaStart(thetaStartSEXP);
    Rcpp::traits::input_parameter< List >::type panelDeltaStart(panelDeltaStartSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type deltaFamilyMeanStart(deltaFamilyMeanStartSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type deltaFamilyVarianceStart(deltaFamilyVarianceStartSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<NumericMatrix> >::type deltaDesignMatrix(deltaDesignMatrixSEXP);
    Rcpp::traits::input_parameter< List >::type thinning(thinningSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< bool >::type progress(progressSEXP);
    rcpp_result_gen = Rcpp::wrap(logisticSample(nSamples, burnIn, panelY, panelDesignMatrix, order, distributionNames, priors, samplingSchemes, thetaStart, panelDeltaStart, deltaFamilyMeanStart, deltaFamilyVarianceStart, deltaDesignMatrix, thinning, verbose, progress));
    return rcpp_result_gen;
END_RCPP
}
// logisticSampleY
List logisticSampleY(List panelExplanatoryVariablesR, List panelDeltaSampleR, List panelZ0SampleR, List distributionSampleR, StringVector distributionNames, unsigned int order);
RcppExport SEXP positivemixtures_logisticSampleY(SEXP panelExplanatoryVariablesRSEXP, SEXP panelDeltaSampleRSEXP, SEXP panelZ0SampleRSEXP, SEXP distributionSampleRSEXP, SEXP distributionNamesSEXP, SEXP orderSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type panelExplanatoryVariablesR(panelExplanatoryVariablesRSEXP);
    Rcpp::traits::input_parameter< List >::type panelDeltaSampleR(panelDeltaSampleRSEXP);
    Rcpp::traits::input_parameter< List >::type panelZ0SampleR(panelZ0SampleRSEXP);
    Rcpp::traits::input_parameter< List >::type distributionSampleR(distributionSampleRSEXP);
    Rcpp::traits::input_parameter< StringVector >::type distributionNames(distributionNamesSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type order(orderSEXP);
    rcpp_result_gen = Rcpp::wrap(logisticSampleY(panelExplanatoryVariablesR, panelDeltaSampleR, panelZ0SampleR, distributionSampleR, distributionNames, order));
    return rcpp_result_gen;
END_RCPP
}
// benchmarkLogistic
void benchmarkLogistic(unsigned int nDeltas, unsigned int nValues, unsigned int nIterations);
RcppExport SEXP positivemixtures_benchmarkLogistic(SEXP nDeltasSEXP, SEXP nValuesSEXP, SEXP nIterationsSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type nDeltas(nDeltasSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nValues(nValuesSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nIterations(nIterationsSEXP);
    benchmarkLogistic(nDeltas, nValues, nIterations);
    return R_NilValue;
END_RCPP
}
// rpolyagammaVector
NumericVector rpolyagammaVector(unsigned int length, unsigned int n, double z);
RcppExport SEXP positivemixtures_rpolyagammaVector(SEXP lengthSEXP, SEXP nSEXP, SEXP zSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type length(lengthSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type n(nSEXP);
    Rcpp::traits::input_parameter< double >::type z(zSEXP);
    rcpp_result_gen = Rcpp::wrap(rpolyagammaVector(length, n, z));
    return rcpp_result_gen;
END_RCPP
}
// thinplateBasis2d
List thinplateBasis2d(NumericMatrix X, unsigned int nBases);
RcppExport SEXP positivemixtures_thinplateBasis2d(SEXP XSEXP, SEXP nBasesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nBases(nBasesSEXP);
    rcpp_result_gen = Rcpp::wrap(thinplateBasis2d(X, nBases));
    return rcpp_result_gen;
END_RCPP
}
// levelsToListIntegerVector
List levelsToListIntegerVector(IntegerVector x, IntegerVector xLevels);
RcppExport SEXP positivemixtures_levelsToListIntegerVector(SEXP xSEXP, SEXP xLevelsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< IntegerVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type xLevels(xLevelsSEXP);
    rcpp_result_gen = Rcpp::wrap(levelsToListIntegerVector(x, xLevels));
    return rcpp_result_gen;
END_RCPP
}
// levelsToListNumericVector
List levelsToListNumericVector(NumericVector x, IntegerVector xLevels);
RcppExport SEXP positivemixtures_levelsToListNumericVector(SEXP xSEXP, SEXP xLevelsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type xLevels(xLevelsSEXP);
    rcpp_result_gen = Rcpp::wrap(levelsToListNumericVector(x, xLevels));
    return rcpp_result_gen;
END_RCPP
}
// levelsToListNumericMatrix
List levelsToListNumericMatrix(NumericMatrix x, IntegerVector xLevels);
RcppExport SEXP positivemixtures_levelsToListNumericMatrix(SEXP xSEXP, SEXP xLevelsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type x(xSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type xLevels(xLevelsSEXP);
    rcpp_result_gen = Rcpp::wrap(levelsToListNumericMatrix(x, xLevels));
    return rcpp_result_gen;
END_RCPP
}
