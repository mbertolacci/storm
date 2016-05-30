.distribution_parameter_names <- list(
    gamma=c('alpha', 'beta'),
    gev=c('mu', 'sigma', 'xi'),
    gengamma=c('mu', 'sigma', 'Q')
)

.distribution_n_vars <- list(
    gamma=2,
    gengamma=3,
    gev=3
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

.default_prior <- function(distributions) {
    lapply(distributions, function(distribution) {
        if (distribution == 'gamma') {
            return(list(type='uniform', bounds=matrix(c(0, 0, 1000, 1000), nrow=2)))
        } else if (distribution == 'gengamma') {
            return(list(type='uniform', bounds=matrix(c(-1000, 0, -100, 1000, 1000, 100), nrow=3)))
        } else if (distribution == 'gev') {
            return(list(type='uniform', bounds=matrix(c(0, 0, 0, 1000, 1000, 1000), nrow=3)))
        }
    })
}

.default_sampling_scheme <- function(distributions) {
    lapply(distributions, function(distribution) {
        if (distribution == 'gamma') {
            return(list(use_mle=TRUE, use_observed_information=TRUE, observed_information_inflation_factor=1))
        } else if (distribution == 'gengamma') {
            return(list(use_mle=FALSE, use_observed_information=TRUE, observed_information_inflation_factor=1))
        } else if (distribution == 'gev') {
            return(list(use_mle=FALSE, use_observed_information=FALSE, covariance=diag(rep(1, 3))))
        }
    })
}

.mixture_initial_values <- function(y, distributions) {
    z_start <- rep(1, length(y))

    y_positive_cdf <- ecdf(y[y > 0 & !is.na(y)])

    for (i in 1 : length(y)) {
        if (is.na(y[i])) {
            z_start[i] <- sample.int(3, 1)
        } else if (y[i] == 0) {
            z_start[i] < 1
        } else {
            p <- y_positive_cdf(y[i])
            z_start[i] <- 1 + sample.int(2, 1, prob=c(1 - p, p))
        }
    }

    theta_start <- list()
    for (i in 1 : length(distributions)) {
        y_mixture <- y[z_start == i + 1]
        y_mixture <- y_mixture[!is.na(y_mixture)]
        if (distributions[i] == 'gamma') {
            theta_start <- append(theta_start, list(gamma_mle(y_mixture)))
        } else if (distributions[i] == 'gev') {
            theta_start <- append(theta_start, list(gevPwmEstimateConstrained(y_mixture, 0)))
        } else if (distributions[i] == 'gengamma') {
            results <- gengamma_mle(y_mixture)
            if (results$convergence != 0) {
                print(results)
                stop('gengamma mle did not converge')
            }
            theta_start <- append(theta_start, list(results$par))
        }
    }

    return (list(z_start=z_start, theta_start=theta_start))
}
