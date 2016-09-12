#' Compute the density function for the GEV
#' @export
dgev <- function(x, mu, sigma, xi, log = FALSE) {
    if (xi == 0) {
        t <- (x - mu) / sigma
        log_density <- log(1 / sigma) - t - exp(-t)
    } else {
        t <- 1 + xi * (x - mu) / sigma
        log_density <- log(1 / sigma) - (1 + 1 / xi) * log(t) - t ^ (-1 / xi)
    }

    if (log) {
        return(log_density)
    }
    return(exp(log_density))
}

gev_loggradient <- function(x, mu, sigma, xi) {
    t <- 1 + xi * (x - mu) / sigma
    t_inv_xi <- t ^ (-1 / xi)

    t_mu <- -xi / sigma
    t_sigma <- - (xi / sigma ^ 2) * (x - mu)
    t_xi <- (x - mu) / sigma

    ell_mu <- sum(
        - (1 + 1 / xi) * t_mu / t
        + (1 / xi) * t_mu * t_inv_xi / t
    )
    ell_sigma <- -length(x) / sigma + sum(
        - (1 + 1 / xi) * t_sigma / t
        + (1 / xi) * t_sigma * t_inv_xi / t
    )
    ell_xi <- sum(
        (1 / xi ^ 2) * log(t)
        - (1 + 1 / xi) * t_xi / t
        - t_inv_xi * (
            (1 / xi ^ 2) * log(t)
            - (1 / xi) * t_xi / t
        )
    )

    return(c(ell_mu, ell_sigma, ell_xi))
}

gev_llhessian <- function(x, mu, sigma, xi) {
    t <- 1 + xi * (x - mu) / sigma
    t_inv_xi <- t ^ (-1 / xi)

    t_mu <- -xi / sigma
    t_sigma <- - (xi / sigma ^ 2) * (x - mu)
    t_xi <- (x - mu) / sigma

    t_mu_sigma <- xi / sigma ^ 2
    t_mu_xi <- -1 / sigma
    t_sigma_sigma <- ( (2 * xi) / sigma ^ 3) * (x - mu)
    t_sigma_xi <- (-1 / sigma ^ 2) * (x - mu)

    ell_mu_mu <- - (1 + 1 / xi) * sum(
        (t_mu ^ 2 / t ^ 2) * (-1 + (1 / xi) * t_inv_xi)
    )
    ell_mu_sigma <- sum(
        - (1 + 1 / xi) * (t_mu_sigma / t - t_mu * t_sigma / t ^ 2)
        + (1 / xi) * (t_mu_sigma * t_inv_xi / t - (1 + 1 / xi) * t_mu * t_sigma * t_inv_xi / t ^ 2)
    )
    ell_mu_xi <- sum(
        (1 / xi ^ 2) * (t_mu / t) * (1 - t_inv_xi)
        - (1 + 1 / xi) * (t_mu_xi / t - t_mu * t_xi / t ^ 2)
        + (1 / xi) * (
            t_mu_xi * t_inv_xi / t
            + t_mu * (t_inv_xi / t) * (
                (1 / xi ^ 2) * log(t)
                - (1 + 1 / xi) * t_xi / t
            )
        )
    )
    ell_sigma_sigma <- length(x) / sigma ^ 2 + sum(
        - (1 + 1 / xi) * (t_sigma_sigma / t - t_sigma ^ 2 / t ^ 2)
        + (1 / xi) * (t_sigma_sigma * t_inv_xi / t - (1 + 1 / xi) * t_sigma ^ 2 * t_inv_xi / t ^ 2)
    )
    ell_sigma_xi <- sum(
        (1 / xi ^ 2) * (t_sigma / t - t_sigma * t_inv_xi / t)
        - (1 + 1 / xi) * (t_sigma_xi / t - t_sigma * t_xi / t ^ 2)
        + (1 / xi) * (
            t_sigma_xi * t_inv_xi / t
            + t_sigma * (t_inv_xi / t) * (
                (1 / xi ^ 2) * log(t)
                - (1 + 1 / xi) * t_xi / t
            )
        )
    )
    ell_xi_xi <- sum(
        (-2 / xi ^ 3) * log(t) + (1 / xi ^ 2) * t_xi / t
        + (1 / xi ^ 2) * t_xi / t
        + (1 + 1 / xi) * t_xi ^ 2 / t ^ 2
        - t_inv_xi * ( (1 / xi ^ 2) * log(t) - (1 / xi) * t_xi / t) ^ 2
        - t_inv_xi * (
            - (2 / xi ^ 3) * log(t)
            + (1 / xi ^ 2) * t_xi / t
            + (1 / xi ^ 2) * t_xi / t
            + (1 / xi) * t_xi ^ 2 / t ^ 2
        )
    )
    hessian <- rbind(
        c(ell_mu_mu, ell_mu_sigma, ell_mu_xi),
        c(ell_mu_sigma, ell_sigma_sigma, ell_sigma_xi),
        c(ell_mu_xi, ell_sigma_xi, ell_xi_xi)
    )

    return(hessian)
}

#' Find the ML estimate of the parameters for the GEV
#' @export
fgev <- function(x) {
    neg_ell <- function(params) {
        if (params[2] <= 0) {
            return(Inf)
        }
        t <- 1 + params[3] * (x - params[1]) / params[2]
        if (any(t <= 0)) {
            return(Inf)
        }
        return(-sum(log(dgev(x, params[1], params[2], params[3]))))
    }

    sigma_start <- sqrt(6 * var(x)) / pi
    mu_start <- mean(x) - 0.58 * sigma_start
    results <- optim(
        c(mu_start, sigma_start, 0),
        neg_ell
    )

    names(results$par) <- c('mu', 'sigma', 'xi')
    results$value <- -results$value
    return(results)
}

#' Find the constrained ML estimate of the parameters for the GEV
#' @export
fgev_constrained <- function(x, support_min=0) {
    neg_ell <- function(params) {
        mu <- params[1]
        sigma <- params[2]
        xi <- sigma / (mu - support_min)
        if (sigma <= 0) {
            return(Inf)
        }
        t <- 1 + xi * (x - mu) / sigma
        if (any(t <= 0)) {
            return(Inf)
        }
        return(-sum(log(dgev(x, mu, sigma, xi))))
    }

    sigma_start <- sqrt(6 * var(x)) / pi
    mu_start <- mean(x) - 0.58 * sigma_start

    results <- optim(
        c(mu_start, sigma_start),
        neg_ell
    )

    results$par <- c(results$par, results$par[2] / results$par[1])
    names(results$par) <- c('mu', 'sigma', 'xi')
    results$value <- -results$value
    return(results)
}

#' Find the constrained ML estimate of the parameters for the GEV
#' @export
fgev_constrained_loose <- function(x, support_min=0) {
    neg_ell <- function(params) {
        mu <- params[1]
        sigma <- params[2]
        xi <- params[3]
        if (sigma <= 0) {
            return(Inf)
        }
        if (xi <= 0 || support_min > (mu - sigma / xi)) {
            return(Inf)
        }
        t <- 1 + xi * (x - mu) / sigma
        if (any(t <= 0)) {
            return(Inf)
        }
        return(-sum(log(dgev(x, mu, sigma, xi))))
    }

    sigma_start <- sqrt(6 * var(x)) / pi
    mu_start <- mean(x) - 0.58 * sigma_start
    xi_start <- sigma_start / (mu_start - support_min)

    results <- optim(
        c(mu_start, sigma_start, xi_start),
        neg_ell
    )

    names(results$par) <- c('mu', 'sigma', 'xi')
    results$value <- -results$value
    return(results)
}

#' @export
gev_support <- function(mu, sigma, xi) {
    if (xi == 0) {
        return(c(-Inf, Inf))
    } else if (xi > 0) {
        return(c(mu - sigma / xi, Inf))
    } else {
        return(c(-Inf, mu - sigma / xi))
    }
}

gev_support_vector <- function(params) {
    return(gev_support(params[1], params[2], params[3]))
}
