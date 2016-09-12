#' @export
gamma_mle <- function(y, max_iterations = 100, absolute_tolerance = 1e-8) {
    n <- length(y)
    s <- log(sum(y) / n) - sum(log(y)) / n

    alpha_current <- (3 - s + sqrt((s - 3) ^ 2 + 24 * s)) / (12 * s)

    iteration <- 1
    while (iteration < max_iterations) {
        alpha_prev <- alpha_current
        alpha_current <- alpha_prev -
            (log(alpha_prev) - digamma(alpha_prev) - s) / (1 / alpha_prev - trigamma(alpha_prev))

        if (abs(alpha_prev - alpha_current) < absolute_tolerance) {
            break
        }
        iteration <- iteration + 1
    }
    if (iteration == max_iterations) {
        stop('gamma_mle did not converge within max_iterations')
    }

    beta <- (1 / alpha_current) * sum(y) / n

    return(c(alpha = alpha_current, beta = beta))
}
