#' @export
ptsm_independent_sample <- function(
    n_samples,
    y,
    distributions = c('gamma', 'gamma'), prior = NULL,
    burn_in = n_samples, sampling_scheme = NULL,
    starting_values = 'cdf',
    thinning = NULL,
    progress = FALSE
) {
    if (is.null(prior)) {
        prior <- list()
    }
    if (is.null(prior$distributions)) {
        prior$distributions <- .default_distributions_prior(distributions)
    }
    if (is.null(prior$p)) {
        prior$p <- c(1, 1, 1)
    }

    if (is.null(sampling_scheme)) {
        sampling_scheme <- list()
    }
    if (is.null(sampling_scheme$distributions)) {
        sampling_scheme$distributions <- .default_sampling_scheme(distributions)
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
        futile.logger::flog.debug('Getting starting values', name = 'ptsm.logistic_sample')
        starting_values <- .mixture_initial_values(y, distributions, method = starting_values)
    }

    results <- .ptsm_independent_sample(
        n_samples, burn_in, y,
        distributions, prior, sampling_scheme,
        starting_values$z, starting_values$distributions,
        thinning$distributions, thinning$p, thinning$z, thinning$y_missing,
        progress
    )

    if (!is.null(results$sample[['lower']])) {
        results$sample$lower <- coda::mcmc(results$sample$lower, start = 1, thin = thinning$distributions)
        colnames(results$sample$lower) <- .distribution_parameter_names[[distributions[1]]]

        results$sample$upper <- coda::mcmc(results$sample$upper, start = 1, thin = thinning$distributions)
        colnames(results$sample$upper) <- .distribution_parameter_names[[distributions[2]]]
    }
    if (!is.null(results$sample[['p']])) {
        results$sample$p <- coda::mcmc(results$sample$p, start = 1, thin = thinning$p)
        colnames(results$sample$p) <- c('p1', 'p2', 'p3')
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

    return (results)
}
