#' @useDynLib threecomponentmixture
#' @importFrom Rcpp sourceCpp

#' @export
threeComponentMixtureHMMSample <- function(n, burn.in, prior, y) {
    start.time <- proc.time()

    starting.values <- .threeComponentMixtureInitialValues(y)

    results <- .threeComponentMixtureHMMSample(
        n, burn.in, y, prior, starting.values$z.start, starting.values$theta.start
    )
    colnames(results$theta.sample) <- c(
        'alpha',
        'beta',
        'mu',
        'sigma',
        'xi',
        'p11', 'p12', 'p13',
        'p21', 'p22', 'p23',
        'p31', 'p32', 'p33'
    )

    time.taken <- proc.time() - start.time
    cat('Iterations =', n, 'burn in =', burn.in, '\n')
    cat('Simulation time:\n')
    print(time.taken)

    return (results)
}
