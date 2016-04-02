#' @export
threeComponentMixtureHMMAnalyse <- function(y, results, pdf.filename) {
    cat('Estimated membership of first observation:', mean(results$z.sample[, 1]), '\n')
    estimate <- threeComponentMixtureEstimateAverages(results)
    p.stationary.estimate <- threeComponentMixtureEstimateHMMStationaryDistribution(estimate)

    cat('Parameter estimates based on ergodic averages:\n')
    print(estimate)
    cat('Estimated stationary distribution:\n')
    print(p.stationary.estimate)

    cat('Covariance matrix:\n')
    print(cov(results$theta.sample[, 1 : 5]))
    cat('Standard deviations:\n')
    print(sqrt(diag(cov(results$theta.sample[, 1 : 5]))))

    ggplot2::theme_set(ggplot2::theme_bw())

    pdf(file=pdf.filename, paper='a4r', width=11.7, height=8.26)
    threeComponentMixtureHMMPrintDiagnostics(y, results)
    invisible(dev.off())
}
