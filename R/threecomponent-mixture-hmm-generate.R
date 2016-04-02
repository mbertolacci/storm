#' @export
threeComponentMixtureHMMGenerate <- function(n, alpha, beta, xi, sigma, mu, p, z.start) {
    z <- rep(z.start, n)
    for (i in 2 : n) {
        z[i] = sample.int(3, 1, prob=p[z[i - 1], ])
    }
    y <- rep(0, n)

    y[z == 1] <- 0
    y[z == 2] <- rgamma(length(which(z == 2)), alpha, scale=beta)
    y[z == 3] <- rgev(length(which(z == 3)), mu, sigma, xi)

    return(data.frame(z, y))
}
