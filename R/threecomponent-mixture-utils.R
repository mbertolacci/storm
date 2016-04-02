.threeComponentMixtureInitialValues <- function(y) {

    z.start <- rep(1, length(y))

    # HACK(mike): nudge the GEV values up
    y.gev.threshold <- median(y[y > 0], na.rm=TRUE)
    # y.gev.threshold <- quantile(y[y > 0], 2 / 3, na.rm=TRUE)
    # y.gev.threshold <- 0
    z.start[y > y.gev.threshold] <- 3
    z.start[y <= y.gev.threshold] <- 2

    # Sample positive values to be either 2 and 3
    # z.start <- 1 + sample.int(2, length(y > 0), replace=TRUE)
    # All zeroes belong in 1
    z.start[y == 0] <- 1
    # Randomly sample the NAs into any three
    z.start[is.na(y)] <- sample.int(3, length(which(is.na(y))), replace=TRUE)

    y.gamma <- y[z.start == 2]
    mean.gamma <- mean(y.gamma, na.rm=TRUE)
    var.gamma <- var(y.gamma, na.rm=TRUE)
    alpha.start <- mean.gamma^2 / var.gamma
    beta.start <- var.gamma / mean.gamma

    y.gev <- y[z.start == 3]
    # NOTE(mike): constrained to a distribution with support (0, Inf)
    gev.estimates <- gevPwmEstimateConstrained(y.gev[!is.na(y.gev)], 0)
    mu.start <- gev.estimates['mu']
    sigma.start <- gev.estimates['sigma']
    xi.start <- gev.estimates['xi']
    gev.support <- gevSupport(mu.start, sigma.start, xi.start)

    theta.start <- c(alpha.start, beta.start, mu.start, sigma.start, xi.start)
    names(theta.start) <- c('alpha', 'beta', 'mu', 'sigma', 'xi')
    cat('Initial values are\n')
    print(theta.start)

    return (list(z.start=z.start, theta.start=theta.start))
}
