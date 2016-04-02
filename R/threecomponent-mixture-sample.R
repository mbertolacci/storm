#' @useDynLib threecomponentmixture
#' @importFrom Rcpp sourceCpp

#' @export
threeComponentMixtureSample <- function(n, burn.in, prior, y) {
    start.time <- proc.time()

    if (anyNA(y)) {
        stop('Unable to handle missing values')
    }

    starting.values <- .threeComponentMixtureInitialValues(y)

    results <- .threeComponentMixtureSample(
        n, burn.in, y, prior, starting.values$z.start, starting.values$theta.start
    )
    colnames(results$theta.sample) <- c(
        'alpha',
        'beta',
        'mu',
        'sigma',
        'xi',
        'p1',
        'p2',
        'p3'
    )

    time.taken <- proc.time() - start.time
    cat('Iterations =', n, 'burn in =', burn.in, '\n')
    cat('Simulation time:\n')
    print(time.taken)

    return (results)
}
