#' @import ggplot2

#' Print out a slew of useful diagnostics for the given identification
#'
#' @param y The vector of data used to fit the results
#' @param results A list produced by \code{threeComponentMixtureSample} or one of the identification routines
#' @return Nothing
#' @export
threeComponentMixtureHMMPrintDiagnostics <- function(y, results) {
    .printColumnsDiagnostics(results$theta.sample, c(1))
    .printColumnsDiagnostics(results$theta.sample, c(2))
    .printColumnsDiagnostics(results$theta.sample, c(3))
    .printColumnsDiagnostics(results$theta.sample, c(4))
    .printColumnsDiagnostics(results$theta.sample, c(5))
    .printColumnsDiagnostics(results$theta.sample, c(6, 7, 8))
    .printColumnsDiagnostics(results$theta.sample, c(9, 10, 11))
    .printColumnsDiagnostics(results$theta.sample, c(12, 13, 14))

    print(ggpairs(results$theta.sample[, 1 : 5]))

    .printSampleDensity(y, results$y.sample)
    .printZPoints(y, results)

    y.with.missing <- y
    y.with.missing[is.na(y)] <- colMeans(results$y.missing.sample)
    data.missing <- data.frame(t=1 : length(y), is.missing=is.na(y), y=y.with.missing)
    print(ggplot(data.missing, aes(t, y, colour=factor(is.missing))) + geom_point())
}
