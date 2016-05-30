#' @export
ptsm_logistic_generate <- function(
    data, formula, distributions, component_parameters, order=1,
    delta_family_mean=NULL, delta_family_variance=NULL, delta=NULL,
    panel_variable=NULL, ...
) {
    stopifnot(order >= 0)

    # Split the data into each of its panels
    if (is.null(panel_variable)) {
        data_levels <- factor(rep('dummy', nrow(data)))
    } else {
        data_levels <- data[[panel_variable]]
    }

    n_levels <- nlevels(data_levels)
    explanatory_variables <- model.matrix(formula, model.frame(formula, data, na.action=na.pass), ...)
    panel_explanatory_variables <- .levels_to_list(explanatory_variables, data_levels)

    n_deltas <- ncol(explanatory_variables) + 2 * order

    if (is.null(delta)) {
        stopifnot(!is.null(delta_family_mean))
        stopifnot(!is.null(delta_family_variance))
        stopifnot(nrow(delta_family_mean) == 2)
        stopifnot(nrow(delta_family_variance) == 2)
        stopifnot(ncol(delta_family_mean) == n_deltas)
        stopifnot(ncol(delta_family_variance) == n_deltas)

        delta <- list()
        # No delta's provided, so draw them from the family parameters
        for (level_index in 1 : n_levels) {
            level_delta <- delta_family_mean
            for (i in 1 : length(delta_family_mean)) {
                level_delta[i] <- rnorm(1, delta_family_mean[1], sqrt(delta_family_variance[1]))
            }
            delta <- append(delta, list(level_delta))
        }
    }

    if (!is.list(delta)) {
        delta <- list(delta)
    }

    output_z <- c()
    output_y <- c()
    for (level_index in 1 : n_levels) {
        level_sample <- .ptsm_logistic_generate(
            delta[[level_index]], panel_explanatory_variables[[level_index]],
            component_parameters[[1]], component_parameters[[2]], distributions, order
        )
        output_z <- c(output_z, level_sample$z)
        output_y <- c(output_y, level_sample$y)
    }
    output <- list(
        data=data.frame(z=output_z, y=output_y),
        delta=delta
    )
    if (length(output$delta) == 1) {
        output$delta <- delta[[1]]
    }

    return(output)
}
