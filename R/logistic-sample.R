.get_delta_param_names <- function(variable_names, level_name=NULL) {
    if (is.null(level_name)) {
        param_names <- sapply(variable_names, function(name) {
            paste0(name, ':[1]')
        })
        param_names <- c(param_names, sapply(variable_names, function(name) {
            paste0(name, ':[2]')
        }))
    } else {
        param_names <- sapply(variable_names, function(name) {
            paste0(name, ':', level_name, '[1]')
        })
        param_names <- c(param_names, sapply(variable_names, function(name) {
            paste0(name, ':', level_name, '[2]')
        }))
    }

    return(param_names)
}

#' @export
ptsm_logistic_sample <- function(
    n_samples, burn_in,
    data, formula,
    distributions=c('gamma', 'gamma'), prior=NULL, sampling_scheme=NULL,
    order=1,
    theta_sample_thinning=1, z_sample_thinning=0, y_missing_sample_thinning=0,
    panel_variable=NA, verbose=0,
    ...
) {
    stopifnot(order >= 0 && order <= 1)

    start_time <- proc.time()

    all_y <- data[[all.vars(formula)[1]]]
    starting_values <- .mixture_initial_values(all_y, distributions)
    explanatory_variables <- model.matrix(formula, model.frame(formula, data, na.action=na.pass), ...)

    # Split the data into each of its panels
    if (is.na(panel_variable)) {
        data_levels <- factor(rep('dummy', nrow(data)))
    } else {
        data_levels <- data[[panel_variable]]
    }

    n_levels <- nlevels(data_levels)
    n_deltas <- ncol(explanatory_variables) + 2 * order

    if (is.null(prior)) {
        prior <- .default_prior(distributions)

        if (n_levels > 1) {
            # By default, a panel receives hierarchical priors
            prior$logistic <- list(type='hierarchical', parameters=cbind(
                rep(0, 2 * n_deltas),  # mu
                rep(100, 2 * n_deltas),  # sigma_mu
                rep(1.1, 2 * n_deltas),  # alpha
                rep(1, 2 * n_deltas)  # beta
            ))
        } else {
            prior$logistic <- list(type='normal', parameters=cbind(
                rep(0, 2 * n_deltas),  # mu
                rep(100, 2 * n_deltas)  # sigma
            ))
        }
    }

    if (is.null(sampling_scheme)) {
        sampling_scheme <- .default_sampling_scheme(distributions)
        sampling_scheme$logistic <- rep(list(list()), n_levels)
    }

    panel_z_start <- .levels_to_list(starting_values$z_start, data_levels)
    panel_explanatory_variables <- .levels_to_list(explanatory_variables, data_levels)
    panel_y <- .levels_to_list(all_y, data_levels)
    # Set delta starting values to 0
    panel_delta_start <- rep(list(matrix(0, nrow=2, ncol=n_deltas)), n_levels)
    if (prior$logistic$type == 'hierarchical') {
        delta_family_mean_start <- matrix(0, nrow=2, ncol=n_deltas)
        delta_family_variance_start <- matrix(1, nrow=2, ncol=n_deltas)
    } else {
        delta_family_mean_start <- matrix(
            prior$logistic$parameters[, 1], nrow=2, ncol=n_deltas
        )
        delta_family_variance_start <- matrix(
            prior$logistic$parameters[, 2], nrow=2, ncol=n_deltas
        )
    }

    # Run the sampler
    inner_results <- .ptsm_logistic_sample(
        n_samples, burn_in,
        panel_y, panel_explanatory_variables, order,
        distributions, prior, sampling_scheme,
        panel_z_start, starting_values$theta_start,
        panel_delta_start, delta_family_mean_start, delta_family_variance_start,
        theta_sample_thinning, z_sample_thinning, y_missing_sample_thinning,
        verbose
    )

    # Post-process results for output
    results <- list()
    column_names <- .get_parameter_names(distributions)
    if (order == 0) {
        delta_param_names <- colnames(panel_explanatory_variables[[1]])
    } else {
        delta_param_names <- c(colnames(panel_explanatory_variables[[1]]))
        for (i in 1 : order) {
            delta_param_names <- c(
                delta_param_names,
                paste0('z2(t-', i, ')'),
                paste0('z3(t-', i, ')')
            )
        }
    }

    if (n_levels == 1) {
        column_names <- c(column_names, .get_delta_param_names(delta_param_names))
    } else {
        group_names <- c(levels(data[[panel_variable]]), 'family_mean', 'family_variance')
        for (group_name in group_names) {
            column_names <- c(column_names, .get_delta_param_names(delta_param_names, group_name))
        }
    }
    results$theta_sample <- coda::mcmc(inner_results$theta_sample, start=1, thin=theta_sample_thinning)
    colnames(results$theta_sample) <- column_names

    # Unroll the panel level samples into their flat version
    if (!is.null(inner_results$panel_z_sample)) {
        results$z_sample <- t(.levels_from_list(lapply(inner_results$panel_z_sample, t), data_levels))
        results$z_sample <- coda::mcmc(results$z_sample, start=1, thin=z_sample_thinning)
    }
    if (!is.null(inner_results$panel_z0_sample)) {
        results$z0_sample <- coda::mcmc(inner_results$panel_z0_sample, start=1, thin=z_sample_thinning)
    }
    if (!is.null(inner_results$panel_y_missing_sample) && anyNA(all_y)) {
        results$y_missing_sample <- t(.levels_from_list(lapply(inner_results$panel_y_missing_sample, t), data_levels))
        results$y_missing_sample <- coda::mcmc(results$y_missing_sample, start=1, thin=y_missing_sample_thinning)
    }

    results$distributions <- distributions
    results$prior <- prior
    results$sampling_scheme <- sampling_scheme
    results$panel_variable <- panel_variable

    results$data <- data
    results$explanatory_variables <- explanatory_variables

    if (verbose > 0) {
        time_taken <- proc.time() - start_time
        cat('Samples =', n_samples, 'burn in =', burn_in, '\n')
        cat('Simulation time:\n')
        print(time_taken)
    }

    return (results)
}

#' @export
ptsm_logistic_sample_y <- function(sampler_results, y_sample_thinning=1) {
    if (is.na(sampler_results$panel_variable)) {
        data_levels <- factor(rep('dummy', nrow(sampler_results$data)))
    } else {
        data_levels <- sampler_results$data[[sampler_results$panel_variable]]
    }

    panel_explanatory_variables <- .levels_to_list(sampler_results$explanatory_variables, data_levels)

    n_lower_params <- .distribution_n_vars[[sampler_results$distributions[1]]]
    n_upper_params <- .distribution_n_vars[[sampler_results$distributions[2]]]
    n_params <- n_lower_params + n_upper_params
    n_deltas <- ncol(sampler_results$explanatory_variables) + 2

    theta_sample_thinned <- window(sampler_results$theta_sample, 1, thin=y_sample_thinning)
    panel_delta_sample <- lapply(1 : nlevels(data_levels), function(level_index) {
        theta_sample_thinned[
            ,
            (n_params + (level_index - 1) * 2 * n_deltas + 1) : (n_params + level_index * 2 * n_deltas)
        ]
    })
    theta_lower_sample <- theta_sample_thinned[, 1 : n_lower_params]
    theta_upper_sample <- theta_sample_thinned[, (n_lower_params + 1) : n_params]

    inner_results <- .ptsm_logistic_sample_y(
        panel_explanatory_variables,
        panel_delta_sample,
        theta_lower_sample,
        theta_upper_sample,
        sampler_results$distributions
    )

    output <- list()
    output$y_sample <- t(.levels_from_list(lapply(inner_results$panel_y_sample, t), data_levels))
    output$y_sample <- coda::mcmc(output$y_sample, start=1, thin=y_sample_thinning)
    output$y_sample_z <- t(.levels_from_list(lapply(inner_results$panel_y_sample_z, t), data_levels))
    output$y_sample_z <- coda::mcmc(output$y_sample_z, start=1, thin=y_sample_thinning)

    return(output)
}
