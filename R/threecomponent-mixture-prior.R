#' @export
threeComponentMixturePrior <- function(y=NA, theta.p=NA) {
    if (is.na(theta.p)) {
        theta.p <- rep(1, 3)
    }

    prior <- list()
    prior$theta.p <- theta.p

    return(prior)
}
