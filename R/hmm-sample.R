#' @export
ptsm_hmm_sample <- function(
    n_samples, burn_in, y,
    distributions = c('gamma', 'gamma'), prior = NULL, sampling_scheme = NULL,
    theta_sample_thinning = 1, z_sample_thinning = 0, y_missing_sample_thinning = 0,
    verbose = 0
) {
    start_time <- proc.time()

    if (is.null(prior)) {
        prior <- .default_distributions_prior(distributions)
        prior$P <- matrix(rep(0.5, 9), nrow = 3)
    }

    if (is.null(sampling_scheme)) {
        sampling_scheme <- .default_sampling_scheme(distributions)
    }

    starting_values <- .mixture_initial_values(y, distributions)
    if (verbose > 0) {
        cat('Starting parameter values are:\n')
        print(starting_values$theta_start)
    }

    results <- .ptsm_hmm_sample(
        n_samples, burn_in, y,
        distributions, prior, sampling_scheme,
        starting_values$z_start, starting_values$theta_start,
        theta_sample_thinning, z_sample_thinning, y_missing_sample_thinning,
        verbose
    )

    if (!is.null(results$theta_sample)) {
        colnames(results$theta_sample) <- c(
            .get_parameter_names(distributions),
            'p11', 'p12', 'p13',
            'p21', 'p22', 'p23',
            'p31', 'p32', 'p33'
        )
        results$theta_sample <- coda::mcmc(results$theta_sample, start = 1, thin = theta_sample_thinning)
    }
    if (!is.null(results$z_sample)) {
        results$z0_sample <- coda::mcmc(results$z0_sample, start = 1, thin = z_sample_thinning)
        results$z_sample <- coda::mcmc(results$z_sample, start = 1, thin = z_sample_thinning)
    }
    if (!is.null(results$y_missing_sample)) {
        results$y_missing_sample <- coda::mcmc(results$y_missing_sample, start = 1, thin = y_missing_sample_thinning)
    }

    if (verbose > 0) {
        time_taken <- proc.time() - start_time
        cat('Iterations =', n_samples, 'burn in  = ', burn_in, '\n')
        cat('Simulation time:\n')
        print(time_taken)
    }

    return (results)
}
