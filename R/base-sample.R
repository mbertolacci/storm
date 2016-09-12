.distribution_parameter_names <- list(
    gamma = c('alpha', 'beta'),
    gev = c('mu', 'sigma', 'xi'),
    gengamma = c('mu', 'sigma', 'Q')
)

.distribution_n_vars <- list(
    gamma = 2,
    gengamma = 3,
    gev = 3
)

.get_parameter_names <- function(distributions) {
    lower_names <- .distribution_parameter_names[[distributions[1]]]
    upper_names <- .distribution_parameter_names[[distributions[2]]]

    if (distributions[1] == distributions[2]) {
        lower_names <- sapply(lower_names, function(name) {
            paste0(name, '[1]')
        })
        upper_names <- sapply(upper_names, function(name) {
            paste0(name, '[2]')
        })
    }

    output <- c(lower_names, upper_names)
    names(output) <- NULL
    return(output)
}

.default_distributions_prior <- function(distributions) {
    lapply(distributions, function(distribution) {
        if (distribution == 'gamma') {
            return(list(type = 'uniform', bounds = matrix(c(0, 0, 1000, 1000), nrow = 2)))
        } else if (distribution == 'gengamma') {
            return(list(type = 'uniform', bounds = matrix(c(-1000, 0, -100, 1000, 1000, 100), nrow = 3)))
        } else if (distribution == 'gev') {
            return(list(type = 'uniform', bounds = matrix(c(0, 0, 0, 1000, 1000, 1000), nrow = 3)))
        }
    })
}

.default_sampling_scheme <- function(distributions) {
    lapply(distributions, function(distribution) {
        if (distribution == 'gamma') {
            return(list(use_mle = TRUE, use_observed_information = TRUE, observed_information_inflation_factor = 1))
        } else if (distribution == 'gengamma') {
            return(list(use_mle = FALSE, use_observed_information = TRUE, observed_information_inflation_factor = 1))
        } else if (distribution == 'gev') {
            return(list(use_mle = FALSE, use_observed_information = FALSE, covariance = diag(rep(1, 3))))
        }
    })
}

.mixture_initial_values <- function(y, distributions, method = 'cdf') {
    z_start <- rep(1, length(y))

    n_components <- length(distributions)

    if (method == 'cdf') {
        futile.logger::flog.debug('Calculating empirical cdf')
        y_positive_cdf <- ecdf(y[y > 0 & !is.na(y)])

        futile.logger::flog.debug('Generating z values from cdf')
        for (i in 1 : length(y)) {
            if (is.na(y[i])) {
                z_start[i] <- sample.int(n_components + 1, 1)
            } else if (y[i] == 0) {
                z_start[i] <- 1
            } else {
                p <- y_positive_cdf(y[i])
                z_start[i] <- 1 + sample.int(2, 1, prob = c(1 - p, p))
            }
        }
    } else if (method == 'bins') {
        bins <- quantile(y[y > 0 & !is.na(y)], (1 : (n_components - 1)) / n_components)
        z_start <- rep(n_components + 1, length(y))
        for (i in (n_components - 1) : 1) {
            z_start[y < bins[i]] <- i + 1
        }
        z_start[is.na(y)] <- 1 + sample.int(n_components, sum(is.na(y)), replace = TRUE)
        z_start[y == 0] <- 1
    } else if (method == 'random') {
        futile.logger::flog.debug('Generating random z values')
        z_start <- 1 + sample.int(n_components, length(y), replace = TRUE)
        z_start[y == 0] <- 1
    }

    distributions_start <- list()
    for (i in 1 : n_components) {
        futile.logger::flog.debug('Finding starting values for mixture component %d', i)
        y_mixture <- y[z_start == i + 1]
        y_mixture <- y_mixture[!is.na(y_mixture)]
        if (distributions[i] == 'gamma') {
            distributions_start <- append(distributions_start, list(gamma_mle(y_mixture)))
        } else if (distributions[i] == 'gev') {
            distributions_start <- append(distributions_start, list(fgev_constrained(y_mixture, 0)))
        } else if (distributions[i] == 'gengamma') {
            results <- gengamma_mle(y_mixture)
            if (results$convergence != 0) {
                print(results)
                stop('gengamma mle did not converge')
            }
            distributions_start <- append(distributions_start, list(results$par))
        }
    }
    futile.logger::flog.debug('Starting parameter values are %s', list(distributions_start))

    return (list(z = z_start, distributions = distributions_start))
}
