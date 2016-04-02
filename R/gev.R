#' Compute the density function for the GEV
#' @export
dgev <- function(x, mu, sigma, xi) {
    if (xi == 0) {
        t <- (x - mu) / sigma
        return(
            (1 / sigma) * exp(-t) * exp(-exp(-t))
        )
    } else {
        t <- 1 + xi * (x - mu) / sigma
        return(
            (1 / sigma) * t^(-1 - 1 / xi) * exp(-t^(-1 / xi))
        )
    }
}

ldgev <- function(x, mu, sigma, xi) {
    if (xi == 0) {
        t <- (x - mu) / sigma
        return(
            log(1 / sigma) - t - exp(-t)
        )
    } else {
        t <- 1 + xi * (x - mu) / sigma
        return(
            log(1 / sigma) - (1 + 1 / xi) * t - t^(-1 / xi)
        )
    }
}

#' Find the ML estimate of the parameters for the GEV
#' @export
fgev <- function(x) {
    negLikelihood <- function(params) {
        if (params[2] <= 0) {
            return(Inf)
        }
        t <- 1 + params[3] * (x - params[1]) / params[2]
        if (any(t <= 0)) {
            return(Inf)
        }
        return(-sum(log(dgev(x, params[1], params[2], params[3]))))
    }

    sigma.start <- sqrt(6 * var(x)) / pi
    mu.start <- mean(x) - 0.58 * sigma.start
    results <- optim(
        c(mu.start, sigma.start, 0),
        negLikelihood
    )

    names(results$par) <- c('mu', 'sigma', 'xi')
    results$value <- -results$value
    return(results)
}

#' Find the constrained ML estimate of the parameters for the GEV
#' @export
fgevConstrained <- function(x, supportMin=0) {
    negLikelihood <- function(params) {
        mu <- params[1]
        sigma <- params[2]
        xi <- sigma / (mu - supportMin)
        if (sigma <= 0) {
            return(Inf)
        }
        t <- 1 + xi * (x - mu) / sigma
        if (any(t <= 0)) {
            return(Inf)
        }
        return(-sum(log(dgev(x, mu, sigma, xi))))
    }

    sigma.start <- sqrt(6 * var(x)) / pi
    mu.start <- mean(x) - 0.58 * sigma.start

    results <- optim(
        c(mu.start, sigma.start),
        negLikelihood
    )

    results$par <- c(results$par, results$par[2] / results$par[1])
    names(results$par) <- c('mu', 'sigma', 'xi')
    results$value <- -results$value
    return(results)
}

#' Find the constrained ML estimate of the parameters for the GEV
#' @export
fgevConstrainedLoose <- function(x, supportMin=0) {
    negLikelihood <- function(params) {
        mu <- params[1]
        sigma <- params[2]
        xi <- params[3]
        if (sigma <= 0) {
            return(Inf)
        }
        if (xi <= 0 || supportMin > (mu - sigma / xi)) {
            return(Inf)
        }
        t <- 1 + xi * (x - mu) / sigma
        if (any(t <= 0)) {
            return(Inf)
        }
        return(-sum(log(dgev(x, mu, sigma, xi))))
    }

    sigma.start <- sqrt(6 * var(x)) / pi
    mu.start <- mean(x) - 0.58 * sigma.start
    xi.start <- sigma.start / (mu.start - supportMin)

    results <- optim(
        c(mu.start, sigma.start, xi.start),
        negLikelihood
    )

    names(results$par) <- c('mu', 'sigma', 'xi')
    results$value <- -results$value
    return(results)
}

#' @export
gevSupport <- function(mu, sigma, xi) {
    if (xi == 0) {
        return(c(-Inf, Inf))
    } else if (xi > 0) {
        return(c(mu - sigma / xi, Inf))
    } else {
        return(c(-Inf, mu - sigma / xi))
    }
}

gevSupport.vec <- function(params) {
    return(gevSupport(params[1], params[2], params[3]))
}
