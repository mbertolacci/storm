#' @export
ptsm_independent_generate <- function(n, p, distributions, component_parameters) {
    z <- sample.int(length(distributions) + 1, n, replace = TRUE, prob = p)
    y <- .y_given_z(z, distributions, component_parameters)

    return(data.frame(z, y))
}
