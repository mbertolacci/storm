#' @export
ptsm_logistic_ergodic_p <- function(
    delta_samples, z_samples, z_levels, z0_samples, design_matrix, panel_variable, order = 1
) {
    p_hat <- data.frame(
        p1 = rep(0, nrow(design_matrix)),
        p2 = rep(0, nrow(design_matrix)),
        p3 = rep(0, nrow(design_matrix))
    )

    p_hat[[panel_variable]] <- design_matrix[[panel_variable]]

    delta_level_column <- which(colnames(delta_samples) == panel_variable)
    design_matrix_level_column <- which(colnames(design_matrix) == panel_variable)

    for (level_index in 1 : nlevels(delta_samples[[panel_variable]]))  {
        level <- levels(delta_samples[[panel_variable]])[level_index]
        level_indices <- design_matrix[[panel_variable]] == level

        level_design_matrix <- data.matrix(
            design_matrix[level_indices, -design_matrix_level_column]
        )
        level_delta_samples <- data.matrix(
            delta_samples[delta_samples[[panel_variable]] == level, -delta_level_column]
        )
        level_z_samples <- z_samples[, z_levels == level]
        level_z0_samples <- z0_samples[, level_index]

        p_hat[level_indices, c('p1', 'p2', 'p3')] <- .ptsm_logistic_ergodic_p(
            level_delta_samples, level_z_samples, level_z0_samples, level_design_matrix, order
        )
    }

    return(p_hat)
}

.prepare_level_input <- function(delta_samples, z0_samples, design_matrix, data_levels) {
    level_input <- list()
    for (level_index in 1 : nlevels(data_levels))  {
        level <- levels(data_levels)[level_index]
        level_indices <- data_levels == level

        level_input <- c(level_input, list(list(
            design_matrix = design_matrix[level_indices, ],
            delta = delta_samples[, , level_index, ],
            z0 = z0_samples[, level_index]
        )))
    }

    return(level_input)
}

#' @export
ptsm_logistic_predicted_p <- function(delta_samples, z0_samples, design_matrix, data_levels = NULL, order = 1) {
    if (is.null(data_levels)) {
        data_levels <- as.factor(rep('dummy', nrow(design_matrix)))
        dim(delta_samples) <- c(dim(delta_samples)[1], dim(delta_samples)[2], 1, dim(delta_samples)[3])
        z0_samples <- as.matrix(z0_samples)
    }

    stopifnot(length(dim(delta_samples)) == 4)
    stopifnot(dim(delta_samples)[3] == nlevels(data_levels))

    stopifnot(length(dim(z0_samples)) == 2)
    stopifnot(ncol(z0_samples) == nlevels(data_levels))

    level_input <- .prepare_level_input(delta_samples, z0_samples, design_matrix, data_levels)
    p_results <- .ptsm_logistic_predicted_p(level_input, order)

    n_components <- dim(delta_samples)[1] + 1
    p_mat <- matrix(0, nrow = nrow(design_matrix), ncol = n_components)
    p_hat <- data.frame(p_mat)
    colnames(p_hat) <- paste0('p', 1 : n_components)

    for (level_index in 1 : nlevels(data_levels))  {
        level_indices <- data_levels == levels(data_levels)[level_index]
        p_hat[level_indices, ] <- p_results[[level_index]]
    }

    return(p_hat)
}

#' @export
ptsm_logistic_moments <- function(
    distribution_samples, delta_samples, z0_samples,
    design_matrix, condition_on_positive = FALSE, distributions = c('gamma', 'gamma'), data_levels = NULL, order = 1
) {
    stopifnot(identical(distributions, c('gamma', 'gamma')))

    if (is.null(data_levels)) {
        data_levels <- as.factor(rep('dummy', nrow(design_matrix)))
        dim(delta_samples) <- c(dim(delta_samples)[1], dim(delta_samples)[2], 1, dim(delta_samples)[3])
        z0_samples <- as.matrix(z0_samples)
    }

    stopifnot(length(dim(delta_samples)) == 4)
    stopifnot(dim(delta_samples)[3] == nlevels(data_levels))

    stopifnot(length(dim(z0_samples)) == 2)
    stopifnot(ncol(z0_samples) == nlevels(data_levels))

    level_input <- .prepare_level_input(delta_samples, z0_samples, design_matrix, data_levels)
    moment_results <- .ptsm_logistic_moments(distribution_samples, level_input, order, condition_on_positive)

    moments <- matrix(0, nrow = nrow(design_matrix), ncol = 3)
    colnames(moments) <- c('mean', 'variance', 'skew')
    for (level_index in 1 : nlevels(data_levels))  {
        level_indices <- data_levels == levels(data_levels)[level_index]
        moments[level_indices, ] <- moment_results[[level_index]]
    }

    return(moments)
}

#' Calculate mean and quantiles of fitted delta values
#' @export
ptsm_logistic_fitted_delta <- function(sampler_results, probs = c(0.025, 0.1, 0.5, 0.9, 0.975)) {
    stopifnot(length(dim(sampler_results$sample$delta_family_mean)) == 4)

    delta_mean_sample <- aperm(sampler_results$sample$delta_family_mean, c(4, 3, 2, 1))
    level_design_matrix <- sampler_results$level_design_matrix

    delta_fitted <- .ptsm_logistic_fitted_delta(delta_mean_sample, level_design_matrix, probs)
    delta_fitted <- aperm(delta_fitted, c(4, 2, 3, 1))
    dimnames(delta_fitted)[[1]] <- dimnames(sampler_results$sample$delta_family_mean)[[1]]
    dimnames(delta_fitted)[[2]] <- dimnames(sampler_results$sample$delta_family_mean)[[2]]
    dimnames(delta_fitted)[[3]] <- c(
        'mean',
        sprintf('q%.2f', 100 * probs)
    )
    dimnames(delta_fitted)[[4]] <- rownames(level_design_matrix)

    return(delta_fitted)
}
