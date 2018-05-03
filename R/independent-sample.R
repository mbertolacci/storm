#' @export
independent_sample <- function(
    n_samples,
    y,
    distributions = c('gamma', 'gamma'), prior = NULL,
    burn_in = n_samples, sampling_scheme = NULL,
    starting_values = 'bins',
    thinning = NULL,
    progress = FALSE
) {
    n_components <- length(distributions)

    if (is.null(prior)) {
        prior <- list()
    }
    if (is.null(prior$p)) {
        prior$p <- rep(1, n_components + 1)
    }

    if (is.null(sampling_scheme)) {
        sampling_scheme <- list()
    }
    if (is.null(sampling_scheme$distributions)) {
        sampling_scheme$distributions <- .default_sampling_scheme(distributions)
    }

    if (is.null(prior$distributions)) {
        prior$distributions <- .default_distributions_prior(distributions, sampling_scheme$distributions)
    }

    if (is.null(thinning)) {
        thinning <- list()
    }
    # Extend the defaults
    thinning <- .extend_list(
        list(distributions = 1, p = 1, z = 0, y_missing = 0),
        thinning
    )

    if (class(starting_values) == 'character') {
        futile.logger::flog.debug('Getting starting values', name = 'ptsm.independent_sample')
        starting_values <- .mixture_initial_values(y, distributions, starting_values = starting_values)
    }

    results <- .independent_sample(
        n_samples, burn_in, y,
        distributions, prior, sampling_scheme,
        starting_values$z, starting_values$distributions,
        thinning$distributions, thinning$p, thinning$z, thinning$y_missing,
        progress
    )

    if (!is.null(results$sample[['distribution']])) {
        for (k in 1 : n_components) {
            results$sample[['distribution']][[k]] <- coda::mcmc(
                results$sample[['distribution']][[k]], start = 1, thin = thinning$distribution
            )
            colnames(results$sample[['distribution']][[k]]) <- .distribution_parameter_names[[distributions[[k]]]]
        }
    }
    if (!is.null(results$sample[['p']])) {
        results$sample$p <- coda::mcmc(results$sample$p, start = 1, thin = thinning$p)
        colnames(results$sample$p) <- sprintf('p[%d]', 1 : (n_components + 1))
    }
    if (!is.null(results$sample[['z']])) {
        results$sample[['z']] <- coda::mcmc(results$sample[['z']], start = 1, thin = thinning$z)
    }
    if (!is.null(results$sample[['y_missing']])) {
        results$sample[['y_missing']] <- coda::mcmc(
            results$sample[['y_missing']],
            start = 1,
            thin = thinning$y_missing
        )
    }

    results$distributions <- distributions
    results$prior <- prior
    results$sampling_scheme <- sampling_scheme

    class(results) <- 'ptsm_independent_mcmc'

    return(results)
}

#' @export
independent_sample_y <- function(results, n_y) {
    n_components <- ncol(results$sample$p) - 1
    n_iterations <- nrow(results$sample$p)
    z_sample <- matrix(0, nrow = n_iterations, ncol = n_y)
    y_sample <- matrix(0, nrow = n_iterations, ncol = n_y)
    for (iteration in 1 : n_iterations) {
        z_sample[iteration, ] <- sample.int(
            n_components + 1,
            n_y,
            replace = TRUE,
            prob = results$sample$p[iteration, ]
        )
        y_sample[iteration, ] <- .y_given_z(
            z_sample[iteration, ],
            results$distributions,
            lapply(results$sample$distribution, function(x) x[iteration, ])
        )
    }

    list(
        z = if (n_y == 1) as.vector(z_sample) else z_sample,
        y = if (n_y == 1) as.vector(y_sample) else y_sample
    )
}

#' @export
summary.ptsm_independent_mcmc <- function(x) {
    structure(list(
        distribution = lapply(x$sample$distribution, summary),
        p = summary(x$sample$p)
    ), class = 'summary.ptsm_independent_mcmc')
}

#' @export
print.summary.ptsm_independent_mcmc <- function(x) {
    for (i in 1 : length(x[['distribution']])) {
        cat('--- Component', i, 'parameters\n')
        print(x[['distribution']][[i]])
    }
    cat('--- p (mixture weights)\n')
    print(x[['p']])
}

#' @export
window.ptsm_independent_mcmc <- function(x, ...) {
    for (k in 1 : length(x[['sample']][['distribution']])) {
        x[['sample']][['distribution']][[k]] <- window(x[['sample']][['distribution']][[k]], ...)
    }
    x[['sample']][['p']] <- window(x[['sample']][['p']], ...)
    return(x)
}
