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

.get_logistic_sample_prior <- function(prior, distributions, n_levels, n_deltas, n_level_vars, sampling_scheme) {
    if (is.null(prior)) {
        prior <- list()
    }

    if (is.null(prior$distributions)) {
        prior$distributions <- .default_distributions_prior(distributions, sampling_scheme$distributions)
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
logistic_sample <- function(
    n_samples, burn_in,
    data, formula,
    distributions = c('gamma', 'gamma'), prior = NULL, sampling_scheme = NULL,
    starting_values = 'cdf',
    order = 1,
    thinning = NULL, attach_data = TRUE, attach_level_data = TRUE,
    panel_variable = NULL,
    level_data = NULL, level_formula = NULL,
    verbose = 0, progress = FALSE,
    mpi = FALSE, checkpoints = 0, checkpoint_path = 'checkpoints',
    num_threads = 0,
    ...
) {
    use_checkpoints <- checkpoints > 0
    if (use_checkpoints && mpi) {
        checkpoint_path <- file.path(checkpoint_path, Rmpi::mpi.comm.rank(0))
    }

    configuration_path <- file.path(checkpoint_path, 'configuration.rds')
    last_checkpoint_path <- file.path(checkpoint_path, 'last_checkpoint.rds')
    new_checkpoint_path <- file.path(checkpoint_path, 'new_checkpoint.rds')
    if (use_checkpoints && file.exists(configuration_path) && file.exists(last_checkpoint_path)) {
        futile.logger::flog.debug('Found existing configuration', name = 'ptsm.logistic_sample')
        configuration <- readRDS(configuration_path)
        current_checkpoint <- readRDS(last_checkpoint_path)
    } else {
        starting_configuration <- .logistic_starting_configuration(
            data, formula,
            distributions, prior, sampling_scheme,
            starting_values,
            order,
            thinning, attach_data, attach_level_data,
            panel_variable,
            level_data, level_formula,
            verbose, progress,
            mpi, num_threads, ...
        )

        configuration <- starting_configuration$configuration
        current_checkpoint <- list(
            starting_values = starting_configuration$starting_values,
            index = 0,
            burn_in_complete = 0,
            n_samples_complete = 0
        )

        if (use_checkpoints) {
            if (!dir.exists(checkpoint_path)) {
                dir.create(checkpoint_path, recursive = TRUE)
            }
            saveRDS(configuration, configuration_path)
            saveRDS(current_checkpoint, last_checkpoint_path)
        }
    }

    futile.logger::flog.debug('Running sampler', name = 'ptsm.logistic_sample')
    if (!use_checkpoints) {
        final_results <- .logistic_run_sampler(
            n_samples, burn_in,
            configuration,
            current_checkpoint$starting_values
        )
    } else {
        while (current_checkpoint$burn_in_complete < burn_in || current_checkpoint$n_samples_complete < n_samples) {
            this_burn_in <- min(checkpoints, burn_in - current_checkpoint$burn_in_complete)
            this_n_samples <- min(checkpoints - this_burn_in, n_samples - current_checkpoint$n_samples_complete)

            futile.logger::flog.debug(
                'Checkpoint %d. Starting from burn_in %d / %d, n_samples %d / %d',
                current_checkpoint$index,
                current_checkpoint$burn_in_complete,
                burn_in,
                current_checkpoint$n_samples_complete,
                n_samples,
                name = 'ptsm.logistic_sample'
            )

            results <- .logistic_run_sampler(
                this_n_samples, this_burn_in,
                configuration,
                current_checkpoint$starting_values
            )

            if (this_n_samples > 0) {
                saveRDS(results, file.path(checkpoint_path, paste0('checkpoint_results', current_checkpoint$index, '.rds')))
                current_checkpoint$index <- current_checkpoint$index + 1
            }
            current_checkpoint$starting_values <- results$final_values
            current_checkpoint$burn_in_complete <- current_checkpoint$burn_in_complete + this_burn_in
            current_checkpoint$n_samples_complete <- current_checkpoint$n_samples_complete + this_n_samples
            # Save new checkpoint, then (hopefully) atomically move it over the last_checkpoint location
            saveRDS(current_checkpoint, new_checkpoint_path)
            file.rename(new_checkpoint_path, last_checkpoint_path)
        }

        futile.logger::flog.debug('Combining checkpoints', name = 'ptsm.logistic_sample')
        final_results <- NULL
        n_components <- length(configuration$distributions)
        for (index in 0 : (current_checkpoint$index - 1)) {
            futile.logger::flog.trace('Loading checkpoint %d', index, name = 'ptsm.logistic_sample')
            # Flush the results of previous loop iterations
            gc()
            results <- readRDS(file.path(checkpoint_path, paste0('checkpoint_results', index, '.rds')))
            if (is.null(final_results)) {
                final_results <- results
            } else {
                for (k in 1 : n_components) {
                    final_results$sample[['distribution']][[k]] <- rbind(
                        final_results$sample[['distribution']][[k]],
                        results$sample[['distribution']][[k]]
                    )
                }
                final_results$sample[['z0']] <- rbind(
                    final_results$sample[['z0']],
                    results$sample[['z0']]
                )
                final_results$sample[['z']] <- rbind(
                    final_results$sample[['z']],
                    results$sample[['z']]
                )
                final_results$sample[['y_missing']] <- rbind(
                    final_results$sample[['y_missing']],
                    results$sample[['y_missing']]
                )
                final_results$sample[['delta']] <- abind::abind(
                    final_results$sample[['delta']],
                    results$sample[['delta']]
                )
                final_results$sample[['delta_family_mean']] <- abind::abind(
                    final_results$sample[['delta_family_mean']],
                    results$sample[['delta_family_mean']]
                )
                final_results$sample[['delta_family_variance']] <- abind::abind(
                    final_results$sample[['delta_family_variance']],
                    results$sample[['delta_family_variance']]
                )

                final_results$starting_values <- results$starting_values
            }
        }
    }

    return(.logistic_output(configuration, final_results))
}

.logistic_starting_configuration <- function(
    data, formula,
    distributions = c('gamma', 'gamma'), prior = NULL, sampling_scheme = NULL,
    starting_values = 'cdf',
    order = 1,
    thinning = NULL, attach_data = TRUE, attach_level_data = TRUE,
    panel_variable = NULL,
    level_data = NULL, level_formula = NULL,
    verbose = 0, progress = FALSE,
    mpi = FALSE, num_threads = 0, ...
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

        if (nrow(level_data[!is.na(level_data[[panel_variable]]), , drop = FALSE]) != nlevels(level_data[[panel_variable]])) {
            stop('Number of rows in level_data should equal number of levels')
        }

        # Reorder the level data factor to be in the same order as the levels
        na_level_rows <- level_data[is.na(level_data[[panel_variable]]), ]
        level_data <- level_data[match(levels(level_data[[panel_variable]]), level_data[[panel_variable]]), ]
        level_data <- rbind(level_data, na_level_rows)

        n_levels <- nrow(level_data)
        n_missing_levels <- nrow(na_level_rows)
    } else {
        n_levels <- nlevels(data_levels)
        n_missing_levels <- 0
    }

    n_data_levels <- n_levels - n_missing_levels
    n_components <- length(distributions)
    n_deltas <- .get_formula_n_terms(formula) + n_components * order
    n_level_vars <- 1
    if (!is.null(level_formula)) {
        n_level_vars <- .get_formula_n_terms(level_formula)
    }

    if (is.null(sampling_scheme)) {
        sampling_scheme <- list()
    }

    if (is.null(sampling_scheme$distributions)) {
        sampling_scheme$distributions <- .default_sampling_scheme(distributions)
    }
    if (is.null(sampling_scheme$logistic)) {
        sampling_scheme$logistic <- rep(list(list()), n_data_levels)
    }
    stopifnot(.validate_sampling_scheme(sampling_scheme, n_data_levels))

    prior <- .get_logistic_sample_prior(prior, distributions, n_data_levels, n_deltas, n_level_vars, sampling_scheme)

    all_y <- data[[all.vars(formula)[1]]]
    if (!is.list(starting_values) || is.null(starting_values$distributions)) {
        futile.logger::flog.debug('Getting starting values', name = 'ptsm.logistic_sample')
        starting_values <- .mixture_initial_values(all_y, distributions, starting_values)
    }

    futile.logger::flog.debug('Calculating design matrix', name = 'ptsm.logistic_sample')
    design_matrix <- .get_logistic_design_matrix(data, formula, ...)

    futile.logger::flog.debug('Calculating level design matrix', name = 'ptsm.logistic_sample')
    level_design_matrix <- .get_logistic_level_design_matrix(prior, level_data, level_formula, n_levels, mpi)

    futile.logger::flog.trace('Preparing data for sampler', name = 'ptsm.logistic_sample')
    panel_design_matrix <- .levels_to_list(design_matrix, data_levels)
    panel_y <- .levels_to_list(all_y, data_levels)
    if (!is.null(starting_values$delta)) {
        if (is.null(panel_variable) && !is.list(starting_values$delta)) {
            starting_values$delta <- list(starting_values$delta)
        }
    } else {
        # Set delta starting values to 0
        starting_values$delta <- rep(list(matrix(0, nrow = n_components, ncol = n_deltas)), n_levels)
    }

    if (prior$logistic$type == 'hierarchical') {
        if (is.null(starting_values$delta_family_mean)) {
            mean_dim <- c(n_components, n_deltas, ncol(level_design_matrix))
            starting_values$delta_family_mean <- array(
                rnorm(prod(mean_dim)),
                dim = mean_dim
            )
        }

        if (is.null(starting_values$delta_family_variance)) {
            starting_values$delta_family_variance <- matrix(1, nrow = n_components, ncol = n_deltas)
        }
    } else {
        starting_values$delta_family_mean <- prior$logistic$mean$mean
        starting_values$delta_family_variance <- prior$logistic$mean$variance[, , 1]
    }

    if (is.null(thinning)) {
        thinning <- list()
    }
    # Extend the defaults
    thinning <- .extend_list(
        list(distribution = 1, delta = 1, family = 1, z0 = 1, z = 0, y_missing = 0),
        thinning
    )

    return(list(
        configuration = list(
            sampler_arguments = list(
                panel_y, panel_design_matrix, order,
                distributions, prior, sampling_scheme,
                level_design_matrix,
                thinning,
                verbose, progress, num_threads
            ),
            mpi = mpi,
            num_threads = num_threads,
            thinning = thinning,
            attach_data = attach_data,
            attach_level_data = attach_level_data,
            order = order,
            distributions = distributions,
            prior = prior,
            sampling_scheme = sampling_scheme,
            panel_variable = panel_variable,
            data = data,
            design_matrix = design_matrix,
            level_data = level_data,
            level_design_matrix = level_design_matrix
        ),
        starting_values = starting_values
    ))
}

.logistic_run_sampler <- function(n_samples, burn_in, configuration, starting_values) {
    if (configuration$mpi) {
        sample_function <- .logistic_sample_mpi
    } else {
        sample_function <- .logistic_sample
    }

    return(do.call(
        sample_function,
        c(
            list(n_samples, burn_in),
            configuration$sampler_arguments[1 : 6],
            list(
                starting_values$distributions,
                starting_values$delta,
                starting_values$delta_family_mean,
                starting_values$delta_family_variance
            ),
            configuration$sampler_arguments[7 : 11]
        )
    ))
}

.logistic_output <- function(configuration, results) {
    n_components <- length(configuration$distributions)

    # Post-process results for output
    if (configuration$order == 0) {
        delta_param_names <- colnames(configuration$design_matrix)
    } else {
        delta_param_names <- c(colnames(configuration$design_matrix))
        for (i in 1 : configuration$order) {
            delta_param_names <- c(
                delta_param_names,
                paste0('z', 2 : (n_components + 1), '(t-', i, ')')
            )
        }
    }

    if (!is.null(results$sample[['distribution']])) {
        for (k in 1 : n_components) {
            results$sample[['distribution']][[k]] <- coda::mcmc(
                results$sample[['distribution']][[k]], start = 1, thin = configuration$thinning$distribution
            )
            colnames(results$sample[['distribution']][[k]]) <- .distribution_parameter_names[[configuration$distributions[[k]]]]
        }
    }

    if (!is.null(results$sample[['delta']])) {
        if (is.null(configuration$panel_variable)) {
            original_dim <- dim(results$sample[['delta']])
            dim(results$sample[['delta']]) <- original_dim[c(1, 2, 4)]
            results$sample[['delta']] <- provideDimnames(results$sample[['delta']])
            dimnames(results$sample[['delta']])[[1]] <- paste0('k=', 2 : (n_components + 1))
            dimnames(results$sample[['delta']])[[2]] <- delta_param_names
            results$sample[['delta']] <- aperm(results$sample[['delta']], c(3, 1, 2))
        } else {
            results$sample[['delta']] <- provideDimnames(results$sample[['delta']])
            dimnames(results$sample[['delta']])[[1]] <- paste0('k=', 2 : (n_components + 1))
            dimnames(results$sample[['delta']])[[2]] <- delta_param_names
            if (is.null(configuration$level_data)) {
                dimnames(results$sample[['delta']])[[3]] <- levels(configuration$data[[configuration$panel_variable]])
            } else {
                dimnames(results$sample[['delta']])[[3]] <- configuration$level_data[[configuration$panel_variable]]
            }
            dimnames(results$sample[['delta']])[[4]] <- NULL

            results$sample[['delta']] <- aperm(results$sample[['delta']], c(4, 1, 2, 3))
        }
        results$sample[['delta']] <- acoda::mcmca(results$sample[['delta']], start = 1, thin = configuration$thinning$delta)
    }

    if (!is.null(results$sample[['delta_family_mean']]) && configuration$prior$logistic$type == 'hierarchical') {
        results$sample[['delta_family_mean']] <- provideDimnames(results$sample[['delta_family_mean']])
        dimnames(results$sample[['delta_family_mean']])[[1]] <- paste0('k=', 2 : (n_components + 1))
        dimnames(results$sample[['delta_family_mean']])[[2]] <- delta_param_names
        dimnames(results$sample[['delta_family_mean']])[[3]] <- colnames(configuration$level_design_matrix)
        dimnames(results$sample[['delta_family_mean']])[[4]] <- NULL
        results$sample[['delta_family_mean']] <- acoda::mcmca(
            aperm(results$sample[['delta_family_mean']], c(4, 1, 2, 3)),
            start = 1, thin = configuration$thinning$family
        )

        results$sample[['delta_family_variance']] <- provideDimnames(results$sample[['delta_family_variance']])
        dimnames(results$sample[['delta_family_variance']])[[1]] <- paste0('k=', 2 : (n_components + 1))
        dimnames(results$sample[['delta_family_variance']])[[2]] <- delta_param_names
        dimnames(results$sample[['delta_family_variance']])[[3]] <- NULL
        results$sample[['delta_family_variance']] <- acoda::mcmca(
            aperm(results$sample[['delta_family_variance']], c(3, 1, 2)),
            start = 1, thin = configuration$thinning$family
        )
    }

    # Unroll the panel level samples into their flat version
    if (!is.null(results$sample[['z']])) {
        results$sample[['z']] <- t(.levels_from_list(lapply(results$sample[['z']], t), data_levels))
        results$sample[['z']] <- coda::mcmc(results$sample[['z']], start = 1, thin = configuration$thinning[['z']])
    }
    if (!is.null(results$sample[['z0']])) {
        results$sample[['z0']] <- coda::mcmc(results$sample[['z0']], start = 1, thin = configuration$thinning[['z0']])
        if (!is.null(configuration$panel_variable)) {
            colnames(results$sample[['z0']]) <- levels(configuration$data[[configuration$panel_variable]])
        }
    }
    if (!is.null(results$sample[['y_missing']]) && anyNA(all_y)) {
        results$sample[['y_missing']] <- t(.levels_from_list(
            lapply(results$sample[['y_missing']], t),
            data_levels[is.na(all_y)]
        ))
        results$sample[['y_missing']] <- coda::mcmc(results[['y_missing']], start = 1, thin = configuration$thinning$y_missing)
    }

    results$order <- configuration$order
    results$distributions <- configuration$distributions
    results$prior <- configuration$prior
    results$sampling_scheme <- configuration$sampling_scheme
    results$panel_variable <- configuration$panel_variable

    if (configuration$attach_data) {
        results$data <- configuration$data
        results$design_matrix <- configuration$design_matrix
    }

    if (configuration$attach_level_data) {
        results$level_data <- configuration$level_data
        results$level_design_matrix <- configuration$level_design_matrix
    }

    class(results) <- 'ptsmlogistic'

    return(results)
}

#' @export
logistic_sample_y <- function(sampler_results, progress = FALSE) {
    if (is.null(sampler_results$panel_variable)) {
        data_levels <- factor(rep('dummy', nrow(sampler_results$data)))
        original_dim <- dim(sampler_results$sample$delta)

        dim(sampler_results$sample$delta) <- c(original_dim[1], original_dim[2], original_dim[3], 1)
    } else {
        data_levels <- sampler_results$data[[sampler_results$panel_variable]]
    }

    futile.logger::flog.trace('Splitting samples', name = 'ptsm.logistic_sample_y')
    panel_design_matrix <- .levels_to_list(sampler_results$design_matrix, data_levels)
    panel_delta_sample <- lapply(1 : nlevels(data_levels), function(level_index) {
        # aperm puts the iteration index last, which is useful inside the function
        aperm(
            sampler_results$sample$delta[, , , level_index],
            c(2, 3, 1)
        )
    })
    panel_z0_sample <- lapply(1 : nlevels(data_levels), function(level_index) {
        sampler_results$sample$z0[, level_index]
    })

    futile.logger::flog.trace('Sampling values', name = 'ptsm.logistic_sample_y')
    inner_results <- .logistic_sample_y(
        panel_design_matrix,
        panel_delta_sample,
        panel_z0_sample,
        sampler_results$sample$distribution,
        sampler_results$distributions,
        sampler_results$order,
        progress = progress
    )

    futile.logger::flog.trace('Joining y samples', name = 'ptsm.logistic_sample_y')
    output <- list()
    output$y_sample <- .levels_from_list(inner_results$panel_y_sample, data_levels, 1)
    inner_results$panel_y_sample <- NULL
    output$y_sample <- coda::mcmc(
        output$y_sample,
        start = start(sampler_results$sample$delta),
        thin = thin(sampler_results$sample$delta)
    )

    futile.logger::flog.trace('Joining z samples', name = 'ptsm.logistic_sample_y')
    output$y_sample_z <- .levels_from_list(inner_results$panel_y_sample_z, data_levels, 1)
    inner_results$panel_y_sample_z <- NULL
    output$y_sample_z <- coda::mcmc(
        output$y_sample_z,
        start = start(sampler_results$sample$delta),
        thin = thin(sampler_results$sample$delta)
    )

    return(output)
}

#' @export
window.ptsmlogistic <- function(x, ...) {
    for (k in 1 : length(x[['sample']][['distribution']])) {
        x[['sample']][['distribution']][[k]] <- window(x[['sample']][['distribution']][[k]], ...)
    }
    x[['sample']][['delta']] <- window(x[['sample']][['delta']], ...)
    x[['sample']][['z0']] <- window(x[['sample']][['z0']], ...)
    if (!is.null(x[['sample']][['delta_family_mean']])) {
        x[['sample']][['delta_family_mean']] <- window(x[['sample']][['delta_family_mean']], ...)
    }
    if (!is.null(x[['sample']][['delta_family_variance']])) {
        x[['sample']][['delta_family_variance']] <- window(x[['sample']][['delta_family_variance']], ...)
    }
    return(x)
}