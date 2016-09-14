.get_formula_n_terms <- function(formula) {
    formula_terms <- terms(formula)
    return(length(attr(formula_terms, 'order')) + attr(formula_terms, 'intercept'))
}

.validate_prior <- function(prior, n_deltas, n_level_vars) {
    n_components <- length(prior$distributions)

    if (n_components < 2) {
        return(FALSE)
    }

    return(
        all.equal(dim(prior$logistic$mean$mean), c(n_components, n_deltas, n_level_vars)) &&
        all.equal(dim(prior$logistic$mean$variance), c(n_components, n_deltas, n_level_vars)) &&
        all.equal(dim(prior$logistic$variance$alpha), c(n_components, n_deltas)) &&
        all.equal(dim(prior$logistic$variance$beta), c(n_components, n_deltas))
    )
}

.validate_sampling_scheme <- function(sampling_scheme, n_levels) {
    return(length(sampling_scheme$logistic) == n_levels)
}

.get_gaussian_process_basis_design_matrix <- function(input_design_matrix, n_bases, mpi) {
    if (mpi) {
        if (Rmpi::mpi.comm.rank(0) == 0) {
            futile.logger::flog.debug('Gathering design matrices')
        }
        design_matrices <- Rmpi::mpi.gather.Robj(input_design_matrix, root = 0, comm = 0, simplify = FALSE)

        if (Rmpi::mpi.comm.rank(0) == 0) {
            design_matrix <- do.call(rbind, design_matrices)
        }
    } else {
        design_matrix <- input_design_matrix
    }

    if (!mpi || Rmpi::mpi.comm.rank(0) == 0) {
        stopifnot(nrow(design_matrix) >= n_bases)
        gp_design_matrix <- thinplate_basis_2d(design_matrix[, 2 : 3], n_bases)$design_matrix

        colnames(gp_design_matrix) <- c(
            colnames(design_matrix),
            paste0('gp', 1 : n_bases)
        )
    }

    if (mpi) {
        if (Rmpi::mpi.comm.rank(0) == 0) {
            futile.logger::flog.debug('Scattering GP design matrix')

            messages <- list()

            current_row <- 1
            for (rank_index in 1 : length(design_matrices)) {
                rank_length <- nrow(design_matrices[[rank_index]])
                messages[[rank_index]] <- gp_design_matrix[current_row : (current_row + rank_length - 1), ]
                current_row <- current_row + rank_length
            }

            gp_design_matrix <- Rmpi::mpi.scatter.Robj(messages, root = 0, comm = 0)
        } else {
            gp_design_matrix <- Rmpi::mpi.scatter.Robj(NULL, root = 0, comm = 0)
        }

    }

    return(gp_design_matrix)
}

.get_logistic_sample_prior <- function(prior, distributions, n_levels, n_deltas, n_level_vars) {
    if (is.null(prior)) {
        prior <- list()
    }

    if (is.null(prior$distributions)) {
        prior$distributions <- .default_distributions_prior(distributions)
    }

    n_components <- length(prior$distributions)

    if (is.null(prior$logistic)) {
        if (n_levels > 1) {
            prior$logistic <- list(type = 'hierarchical', is_gp = FALSE)
        } else {
            prior$logistic <- list(type = 'normal')
        }
    }

    if (prior$logistic$type == 'hierarchical') {
        if (is.null(prior$logistic$is_gp)) {
            prior$logistic$is_gp <- FALSE
        }

        if (prior$logistic$is_gp) {
            if (is.null(prior$logistic$n_gp_bases)) {
                prior$logistic$n_gp_bases <- min(n_levels, max(10, ceiling(n_levels / 10)))
            }

            if (is.null(prior$logistic$tau_squared)) {
                prior$logistic$tau_squared <- list(
                    alpha = matrix(1.1, nrow = n_components, ncol = n_deltas),
                    beta = matrix(0.5, nrow = n_components, ncol = n_deltas)
                )
            }

            n_level_vars <- n_level_vars + prior$logistic$n_gp_bases
        }
    }

    if (is.null(prior$logistic$mean)) {
        prior$logistic$mean <- list(
            mean = array(0, dim = c(n_components, n_deltas, n_level_vars)),
            variance = array(100, dim = c(n_components, n_deltas, n_level_vars))
        )
    }

    if (is.null(prior$logistic$variance)) {
        prior$logistic$variance <- list(
            alpha = matrix(1.1, nrow = n_components, ncol = n_deltas),
            beta = matrix(0.5, nrow = n_components, ncol = n_deltas)
        )
    }

    stopifnot(.validate_prior(prior, n_deltas, n_level_vars))

    return(prior)
}

.get_logistic_design_matrix <- function(data, formula, ...) {
    return(model.matrix(formula, model.frame(formula, data, na.action = na.pass), ...))
}

.get_logistic_level_design_matrix <- function(prior, level_data, level_formula, n_levels, mpi) {
    if (prior$logistic$type != 'hierarchical') {
        return(NULL)
    }

    # A hierarchical prior means there needs to be a design matrix
    if (is.null(level_data)) {
        # By default, just have intercept terms
        level_design_matrix <- matrix(1, nrow = n_levels, ncol = 1)
        colnames(level_design_matrix) <- c('(Intercept)')
    } else {
        stopifnot(!is.null(level_formula))
        level_design_matrix <- model.matrix(level_formula, model.frame(level_formula, level_data))
    }

    if (prior$logistic$is_gp) {
        # NOTE(mgnb): only support 2 dimensional GP for now
        stopifnot(ncol(level_design_matrix) == 3)

        futile.logger::flog.debug(
            'Using Gaussian Process prior with %d bases',
            prior$logistic$n_gp_bases,
            name = 'ptsm.logistic_sample'
        )

        level_design_matrix <- .get_gaussian_process_basis_design_matrix(
            level_design_matrix, prior$logistic$n_gp_bases, mpi
        )
    }

    return(level_design_matrix)
}

#' @export
ptsm_logistic_sample <- function(
    n_samples, burn_in,
    data, formula,
    distributions = c('gamma', 'gamma'), prior = NULL, sampling_scheme = NULL,
    starting_value_method = 'cdf',
    order = 1,
    thinning = NULL, attach_data = TRUE, attach_level_data = TRUE,
    panel_variable = NULL,
    level_data = NULL, level_formula = NULL,
    verbose = 0, progress = FALSE,
    mpi = FALSE, ...
) {
    stopifnot(order >= 0 && order <= 1)

    if (mpi && Rmpi::mpi.comm.rank(0) != 0) {
        # Only show progress on root node
        progress <- FALSE
    }

    # Get the levels for the data
    if (is.null(panel_variable)) {
        data_levels <- factor(rep('dummy', nrow(data)))
    } else {
        data_levels <- data[[panel_variable]]
    }

    if (!is.null(level_data)) {
        if (!identical(levels(data_levels), levels(level_data[[panel_variable]]))) {
            stop('Levels in data and level_data must be equal')
        }

        if (nrow(level_data) != nlevels(level_data[[panel_variable]])) {
            stop('Number of rows in level_data should equal number of levels')
        }

        # Reorder the level data factor to be in the same order as the levels
        level_data <- level_data[match(levels(level_data[[panel_variable]]), level_data[[panel_variable]]), ]
    }

    n_levels <- nlevels(data_levels)
    n_components <- length(distributions)
    n_deltas <- .get_formula_n_terms(formula) + n_components * order
    n_level_vars <- 1
    if (!is.null(level_formula)) {
        n_level_vars <- .get_formula_n_terms(level_formula)
    }

    prior <- .get_logistic_sample_prior(prior, distributions, n_levels, n_deltas, n_level_vars)

    if (is.null(sampling_scheme)) {
        sampling_scheme <- .default_sampling_scheme(distributions)
    }
    if (is.null(sampling_scheme$logistic)) {
        sampling_scheme$logistic <- rep(list(list()), n_levels)
    }
    stopifnot(.validate_sampling_scheme(sampling_scheme, n_levels))

    futile.logger::flog.debug('Getting starting values', name = 'ptsm.logistic_sample')
    all_y <- data[[all.vars(formula)[1]]]
    starting_values <- .mixture_initial_values(all_y, distributions, method = starting_value_method)

    futile.logger::flog.debug('Calculating design matrix', name = 'ptsm.logistic_sample')
    design_matrix <- .get_logistic_design_matrix(data, formula, ...)

    futile.logger::flog.debug('Calculating level design matrix', name = 'ptsm.logistic_sample')
    level_design_matrix <- .get_logistic_level_design_matrix(prior, level_data, level_formula, n_levels, mpi)

    futile.logger::flog.trace('Preparing data for sampler', name = 'ptsm.logistic_sample')
    panel_z_start <- .levels_to_list(starting_values$z, data_levels)
    panel_z0_start <- rep(1, n_levels)
    panel_design_matrix <- .levels_to_list(design_matrix, data_levels)
    panel_y <- .levels_to_list(all_y, data_levels)
    # Set delta starting values to 0
    panel_delta_start <- rep(list(matrix(0, nrow = n_components, ncol = n_deltas)), n_levels)
    if (prior$logistic$type == 'hierarchical') {
        mean_dim <- c(n_components, n_deltas, ncol(level_design_matrix))
        delta_family_mean_start <- array(
            rnorm(prod(mean_dim)),
            dim = mean_dim
        )
        delta_family_variance_start <- matrix(1, nrow = n_components, ncol = n_deltas)
    } else {
        delta_family_mean_start <- prior$logistic$mean$mean
        delta_family_variance_start <- prior$logistic$mean$variance[, , 1]
    }

    if (is.null(thinning)) {
        thinning <- list()
    }
    # Extend the defaults
    thinning <- .extend_list(
        list(distribution = 1, delta = 1, family = 1, z0 = 1, z = 0, y_missing = 0),
        thinning
    )

    # Run the sampler
    futile.logger::flog.debug('Running sampler', name = 'ptsm.logistic_sample')
    if (mpi) {
        sample_function <- .ptsm_logistic_sample_mpi
    } else {
        sample_function <- .ptsm_logistic_sample
    }

    results <- sample_function(
        n_samples, burn_in,
        panel_y, panel_design_matrix, order,
        distributions, prior, sampling_scheme,
        panel_z_start, panel_z0_start, starting_values$distributions,
        panel_delta_start, delta_family_mean_start, delta_family_variance_start,
        level_design_matrix,
        thinning,
        verbose, progress
    )

    # Post-process results for output
    if (order == 0) {
        delta_param_names <- colnames(design_matrix)
    } else {
        delta_param_names <- c(colnames(design_matrix))
        for (i in 1 : order) {
            delta_param_names <- c(
                delta_param_names,
                paste0('z', 2 : (n_components + 1), '(t-', i, ')')
            )
        }
    }

    if (!is.null(results$sample[['distribution']])) {
        for (k in 1 : n_components) {
            results$sample[['distribution']][[k]] <- coda::mcmc(
                results$sample[['distribution']][[k]], start = 1, thin = thinning$distribution
            )
            colnames(results$sample[['distribution']][[k]]) <- .distribution_parameter_names[[distributions[[k]]]]
        }
    }

    if (!is.null(results$sample[['delta']])) {
        if (is.null(panel_variable)) {
            original_dim <- dim(results$sample[['delta']])
            dim(results$sample[['delta']]) <- original_dim[c(1, 2, 4)]
            results$sample[['delta']] <- provideDimnames(results$sample[['delta']])
            # dimnames(results$sample[['delta']])[[1]] <- c('k=2', 'k=3')
            dimnames(results$sample[['delta']])[[2]] <- delta_param_names
        } else {
            results$sample[['delta']] <- provideDimnames(results$sample[['delta']])
            # dimnames(results$sample[['delta']])[[1]] <- c('k=2', 'k=3')
            dimnames(results$sample[['delta']])[[2]] <- delta_param_names
            dimnames(results$sample[['delta']])[[3]] <- levels(data[[panel_variable]])
            dimnames(results$sample[['delta']])[[4]] <- NULL
        }
    }

    if (!is.null(results$sample[['delta_family_mean']]) && prior$logistic$type == 'hierarchical') {
        results$sample[['delta_family_mean']] <- provideDimnames(results$sample[['delta_family_mean']])
        # dimnames(results$sample[['delta_family_mean']])[[1]] <- c('k=2', 'k=3')
        dimnames(results$sample[['delta_family_mean']])[[2]] <- delta_param_names
        dimnames(results$sample[['delta_family_mean']])[[3]] <- colnames(level_design_matrix)
        dimnames(results$sample[['delta_family_mean']])[[4]] <- NULL

        results$sample[['delta_family_variance']] <- provideDimnames(results$sample[['delta_family_variance']])
        # dimnames(results$sample[['delta_family_variance']])[[1]] <- c('k=2', 'k=3')
        dimnames(results$sample[['delta_family_variance']])[[2]] <- delta_param_names
        dimnames(results$sample[['delta_family_variance']])[[3]] <- NULL
    }

    # Unroll the panel level samples into their flat version
    if (!is.null(results$sample[['z']])) {
        results$sample[['z']] <- t(.levels_from_list(lapply(results$sample[['z']], t), data_levels))
        results$sample[['z']] <- coda::mcmc(results$sample[['z']], start = 1, thin = thinning[['z']])
    }
    if (!is.null(results$sample[['z0']])) {
        results$sample[['z0']] <- coda::mcmc(results$sample[['z0']], start = 1, thin = thinning[['z0']])
        if (!is.null(panel_variable)) {
            colnames(results$sample[['z0']]) <- levels(data[[panel_variable]])
        }
    }
    if (!is.null(results$sample[['y_missing']]) && anyNA(all_y)) {
        results$sample[['y_missing']] <- t(.levels_from_list(
            lapply(results$sample[['y_missing']], t),
            data_levels[is.na(all_y)]
        ))
        results$sample[['y_missing']] <- coda::mcmc(results[['y_missing']], start = 1, thin = thinning$y_missing)
    }

    results$distributions <- distributions
    results$prior <- prior
    results$sampling_scheme <- sampling_scheme
    results$panel_variable <- panel_variable

    if (attach_data) {
        results$data <- data
        results$design_matrix <- design_matrix
    }

    if (attach_level_data) {
        results$level_data <- level_data
        results$level_design_matrix <- level_design_matrix
    }

    return (results)
}

#' @export
ptsm_logistic_sample_y <- function(sampler_results, y_sample_thinning = 1) {
    if (is.null(sampler_results$panel_variable)) {
        data_levels <- factor(rep('dummy', nrow(sampler_results$data)))
    } else {
        data_levels <- sampler_results$data[[sampler_results$panel_variable]]
    }

    panel_design_matrix <- .levels_to_list(sampler_results$design_matrix, data_levels)
    panel_delta_sample <- lapply(1 : nlevels(data_levels), function(level_index) {
        # HACK(mgnb): fix this crappy thinning routine
        n_iterations <- dim(sampler_results$sample$delta)[4]
        sampler_results$sample$delta[, , level_index, ((1 : n_iterations) - 1) %% y_sample_thinning == 0]
    })
    panel_z0_sample <- lapply(1 : nlevels(data_levels), function(level_index) {
        window(sampler_results$sample$z0[, level_index], 1, thin = y_sample_thinning)
    })

    thinned_distribution_sample <- list()
    for (k in 1 : length(sampler_results$sample$distribution)) {
        thinned_distribution_sample[[k]] <- window(sampler_results$sample$distribution[[k]], 1, thin = y_sample_thinning)
    }

    inner_results <- .ptsm_logistic_sample_y(
        panel_design_matrix,
        panel_delta_sample,
        panel_z0_sample,
        thinned_distribution_sample,
        sampler_results$distributions
    )

    output <- list()
    output$y_sample <- t(.levels_from_list(lapply(inner_results$panel_y_sample, t), data_levels))
    output$y_sample <- coda::mcmc(output$y_sample, start = 1, thin = y_sample_thinning)
    output$y_sample_z <- t(.levels_from_list(lapply(inner_results$panel_y_sample_z, t), data_levels))
    output$y_sample_z <- coda::mcmc(output$y_sample_z, start = 1, thin = y_sample_thinning)

    return(output)
}
