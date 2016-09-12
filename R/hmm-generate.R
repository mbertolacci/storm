#' @export
ptsm_hmm_generate <- function(n, P, distributions, component_parameters, z0 = 1) {
    z <- rep(z0, n + 1)
    for (i in 2 : n) {
        z[i] <- sample.int(3, 1, prob = P[z[i - 1], ])
    }
    z <- z[-1]
    y <- .y_given_z(z, distributions, component_parameters)

    return(data.frame(z, y))
}
