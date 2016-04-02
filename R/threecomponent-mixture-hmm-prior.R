#' @export
threeComponentMixtureHMMPrior <- function(y=NA, theta.p=NA) {
    if (is.na(theta.p)) {
        theta.p <- matrix(rep(1, 9), nrow=3, ncol=3)
    }

    prior <- list()
    prior$theta.p <- theta.p

    return(prior)
}
