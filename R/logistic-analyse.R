#' Take a wide format theta_samples and perform a melt-like operation on it to split it into a delta for each level
#' @export
ptsm_logistic_melt_delta_samples <- function(theta_samples, levels, distributions, level_var='level') {
    n_distribution_params <- (
        .distribution_n_vars[[distributions[1]]] + .distribution_n_vars[[distributions[2]]]
    )

    levels <- c(sort(levels), 'family')
    # The plus 1 below is because there are also the family variances at the end
    n_deltas <- (ncol(theta_samples) - n_distribution_params) / (length(levels) + 1)

    delta_samples <- theta_samples[, (n_distribution_params + 1) : (n_distribution_params + length(levels) * n_deltas)]

    delta_samples_melted <- matrix(0, nrow=0, ncol=n_deltas)
    for (level_index in 1 : length(levels)) {
        level_delta_samples <- delta_samples[, (1 + (level_index - 1) * n_deltas) : (level_index * n_deltas)]
        delta_samples_melted <- rbind(delta_samples_melted, level_delta_samples)
    }
    delta_samples_melted <- data.frame(delta_samples_melted)
    delta_samples_melted[[level_var]] <- factor(sapply(levels, function(level) {
        rep(level, nrow(delta_samples))
    }))

    base_colnames <- colnames(delta_samples)[1 : n_deltas]
    colnames(delta_samples_melted) <- c(sub(':[^\\[]+', '', base_colnames), level_var)

    return(delta_samples_melted[, c(n_deltas + 1, 1 : n_deltas)])
}

#' @export
ptsm_logistic_ergodic_p <- function(
    delta_samples, z_samples, z_levels, z0_samples, explanatory_variables, panel_variable=NA, order=1
) {
    p_hat <- data.frame(
        p1=rep(0, nrow(explanatory_variables)),
        p2=rep(0, nrow(explanatory_variables)),
        p3=rep(0, nrow(explanatory_variables))
    )
    p_hat[[panel_variable]] <- explanatory_variables[[panel_variable]]

    delta_level_column <- which(colnames(delta_samples) == panel_variable)
    explanatory_variables_level_column <- which(colnames(explanatory_variables) == panel_variable)

    for (level_index in 1 : nlevels(delta_samples[[panel_variable]]))  {
        level <- levels(delta_samples[[panel_variable]])[level_index]
        level_indices <- explanatory_variables[[panel_variable]] == level

        level_explanatory_variables <- data.matrix(
            explanatory_variables[level_indices, -explanatory_variables_level_column]
        )
        level_delta_samples <- data.matrix(
            delta_samples[delta_samples[[panel_variable]] == level, -delta_level_column]
        )
        level_z_samples <- z_samples[, z_levels == level]
        level_z0_samples <- z0_samples[, level_index]

        p_hat[level_indices, c('p1', 'p2', 'p3')] <- .ptsm_logistic_ergodic_p(
            level_delta_samples, level_z_samples, level_z0_samples, level_explanatory_variables, order
        )
    }

    return(p_hat)
}

#' @export
ptsm_logistic_predicted_p <- function(delta_samples, z0_samples, explanatory_variables, panel_variable=NA, order=1) {
    p_hat <- data.frame(
        p1=rep(0, nrow(explanatory_variables)),
        p2=rep(0, nrow(explanatory_variables)),
        p3=rep(0, nrow(explanatory_variables))
    )
    p_hat[[panel_variable]] <- explanatory_variables[[panel_variable]]

    delta_level_column <- which(colnames(delta_samples) == panel_variable)
    explanatory_variables_level_column <- which(colnames(explanatory_variables) == panel_variable)

    for (level_index in 1 : nlevels(delta_samples[[panel_variable]]))  {
        level <- levels(delta_samples[[panel_variable]])[level_index]
        level_indices <- explanatory_variables[[panel_variable]] == level

        level_explanatory_variables <- data.matrix(
            explanatory_variables[level_indices, -explanatory_variables_level_column]
        )
        level_delta_samples <- data.matrix(
            delta_samples[delta_samples[[panel_variable]] == level, -delta_level_column]
        )
        level_z0_samples <- z0_samples[, level_index]

        p_hat[level_indices, c('p1', 'p2', 'p3')] <- .ptsm_logistic_predicted_p(
            level_delta_samples, level_z0_samples, level_explanatory_variables, order
        )
    }

    return(p_hat)
}
