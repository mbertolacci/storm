#' @export
dgengamma <- function(x, mu, sigma, Q, log = FALSE) {
    if (Q == 0) {
        log_density <- dlnorm(x, mu, sigma, log = TRUE)
    } else {
        w <- (log(x) - mu) / sigma

        Q_inv_squared <- 1 / Q ^ 2

        log_density <- (
            -log(sigma * x)
            + log(abs(Q))
            + Q_inv_squared * log(Q_inv_squared)
            + Q_inv_squared * (Q * w - exp(Q * w))
            - lgamma(Q_inv_squared)
        )
    }

    if (log) {
        return(log_density)
    }
    return(exp(log_density))
}

#' @export
rgengamma <- function(n, mu, sigma, Q) {
    if (Q == 0) {
        return(rlnorm(n, mu, sigma))
    } else {
        w <- log(Q ^ 2 * rgamma(n, 1 / Q ^ 2, 1)) / Q
        return(exp(mu + sigma * w))
    }
}

gengamma_llgradient <- function(x, mu, sigma, Q) {
    if (Q == 0) {
        return(NaN)
    }
    n <- length(x)
    w <- (log(x) - mu) / sigma
    sum_log_x <- sum(log(x))
    sum_exp_Qw <- sum(exp(Q * w))
    sum_w_exp_Qw <- sum(w * exp(Q * w))

    gradient <- c(0, 0, 0)
    gradient[1] <- -n / (sigma * Q) + sum_exp_Qw / (sigma * Q)
    gradient[2] <- (
        n * (-1 / sigma + mu / (sigma ^ 2 * Q))
        - sum_log_x / (sigma ^ 2 * Q)
        + sum_w_exp_Qw / (sigma * Q)
    )
    gradient[3] <- (
        n * (
            1 / Q
            + 2 * log(Q ^ 2) / (Q ^ 3)
            - 2 / (Q ^ 3)
            + 2 * digamma(1 / (Q ^ 2)) / (Q ^ 3)
            + mu / (sigma * Q ^ 2)
        )
        - sum_log_x / (sigma * Q ^ 2)
        + 2 * sum_exp_Qw / (Q ^ 3)
        - sum_w_exp_Qw / (Q ^ 2)
    )

    return(gradient)
}

gengamma_llhessian <- function(x, mu, sigma, Q) {
    n <- length(x)
    w <- (log(x) - mu) / sigma
    sum_log_x <- sum(log(x))
    sum_exp_Qw <- sum(exp(Q * w))
    sum_w_exp_Qw <- sum(w * exp(Q * w))
    sum_w_squared_exp_Qw <- sum(w ^ 2 * exp(Q * w))

    hessian <- matrix(0, nrow = 3, ncol = 3)
    hessian[1, 1] <- -sum_exp_Qw / (sigma ^ 2)
    hessian[1, 2] <- n / (sigma ^ 2 * Q) - sum_exp_Qw / (sigma ^ 2 * Q) - sum_w_exp_Qw / (sigma ^ 2)
    hessian[1, 3] <- n / (sigma * Q ^ 2) - sum_exp_Qw / (sigma * Q ^ 2) + sum_w_exp_Qw / (sigma * Q)
    hessian[2, 2] <- (
        n / (sigma ^ 2)
        - 2 * n * mu / (sigma ^ 3 * Q)
        + 2 * sum_log_x / (sigma ^ 3 * Q)
        - 2 * sum_w_exp_Qw / (sigma ^ 2 * Q)
        - sum_w_squared_exp_Qw / (sigma ^ 2)
    )
    hessian[2, 3] <- (
        - n * mu / (sigma ^ 2 * Q ^ 2)
        + sum_log_x / (sigma ^ 2 * Q ^ 2)
        - sum_w_exp_Qw / (sigma * Q ^ 2)
        + sum_w_squared_exp_Qw / (sigma * Q)
    )
    hessian[3, 3] <- (
        n * (
            -1 / (Q ^ 2)
            + 6 * log(1 / (Q ^ 2)) / (Q ^ 4)
            + 10 / (Q ^ 4)
            - 6 * digamma(1 / (Q ^ 2)) / (Q ^ 4)
            - 4 * trigamma(1 / (Q ^ 2)) / (Q ^ 6)
            - 2 * mu / (sigma * Q ^ 3)
        )
        + 2 * sum_log_x / (sigma * Q ^ 3)
        - 6 * sum_exp_Qw / (Q ^ 4)
        + 4 * sum_w_exp_Qw / (Q ^ 3)
        - sum_w_squared_exp_Qw / Q ^ 2
    )

    hessian[2, 1] <- hessian[1, 2]
    hessian[3, 1] <- hessian[1, 3]
    hessian[3, 2] <- hessian[2, 3]

    return(hessian)
}

gengamma_mle <- function(y) {
    neg_log_likelihood <- function(params) {
        -sum(dgengamma(y, params[1], exp(params[2]), params[3], log = TRUE))
    }

    neg_ll_gradient <- function(params) {
        -gengamma_llgradient(y, params[1], exp(params[2]), params[3])
    }

    results <- optim(
        c(mean(y), log(sd(y)), 0.1),
        fn = neg_log_likelihood, gr = neg_ll_gradient,
        method = 'BFGS',
        control = list(maxit = 5000)
    )
    results$par[2] <- exp(results$par[2])
    names(results$par) <- c('mu', 'sigma', 'Q')

    return(results)
}
