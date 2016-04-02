#' @export
threeComponentMixtureEstimateHMMAverages <- function(results) {
    .mixtureEstimateAverages(results)
}

#' @export
threeComponentMixtureEstimateHMMStationaryDistribution <- function(theta.estimate, max.iterations=20, tolerance=1e-6) {
    p <- matrix(
        theta.estimate[6 : 14],
        nrow=3,
        ncol=3,
        byrow=TRUE
    )

    p.stationary <- p
    i <- 0
    while (i < max.iterations) {
        p.stationary.old <- p.stationary
        p.stationary <- p.stationary.old %*% p
        if (max(abs(p.stationary - p.stationary.old)) < tolerance) {
            break;
        }
        i <- i + 1
    }
    if (i == max.iterations) {
        warning('Stationary distribution did not converge within', max.iterations, 'iterations')
    }
    p.stationary <- p.stationary[1, ]
    names(p.stationary) <- c('p1', 'p2', 'p2')
    return(p.stationary)
}

#' @export
threeComponentMixtureEstimateHMMPosteriorDensityMode <- function(results) {
    .mixtureEstimatePosteriorDensityMode(results)
}
