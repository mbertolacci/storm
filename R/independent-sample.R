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
        starting_values <- .mixture_initial_values(y, distributions, method = starting_values)
    }

    results <- .ptsm_independent_sample(
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
        colnames(results$sample$p) <- paste0('p', 1 : (n_components + 1))
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
