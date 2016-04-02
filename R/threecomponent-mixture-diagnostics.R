#' @importFrom GGally ggpairs

#' Print out a slew of useful diagnostics for the given identification
#'
#' @param y The vector of data used to fit the results
#' @param results A list produced by \code{threeComponentMixtureSample} or one of the identification routines
#' @return Nothing
#' @export
threeComponentMixturePrintDiagnostics <- function(y, results) {
    .printColumnsDiagnostics(results$theta.sample, c(1))
    .printColumnsDiagnostics(results$theta.sample, c(2))
    .printColumnsDiagnostics(results$theta.sample, c(3))
    .printColumnsDiagnostics(results$theta.sample, c(4))
    .printColumnsDiagnostics(results$theta.sample, c(5))
    .printColumnsDiagnostics(results$theta.sample, c(6, 7, 8))

    print(ggpairs(results$theta.sample[, 1 : 5]))

    .printSampleDensity(y, results$y.sample)
    .printZPoints(y, results)
}
