#' @export
logistic_generate <- function(
    data, formula, distributions, component_parameters,
    order = 1,
    delta_family_mean = NULL, delta_family_variance = NULL, delta = NULL,
    level_data = NULL,
    level_formula = NULL,
    panel_variable = NULL, ...
) {
    stopifnot(order >= 0)

    # Split the data into each of its panels
    if (is.null(panel_variable)) {
        data_levels <- factor(rep('dummy', nrow(data)))
    } else {
        data_levels <- data[[panel_variable]]
    }

    n_levels <- nlevels(data_levels)
    explanatory_variables <- model.matrix(formula, model.frame(formula, data, na.action = na.pass), ...)
    panel_explanatory_variables <- .levels_to_list(explanatory_variables, data_levels)

    n_deltas <- ncol(explanatory_variables) + length(distributions) * order

    if (is.null(delta)) {
        stopifnot(!is.null(delta_family_mean))
        stopifnot(!is.null(delta_family_variance))
        stopifnot(dim(delta_family_mean)[1] == length(distributions))
        stopifnot(nrow(delta_family_variance) == length(distributions))
        stopifnot(ncol(delta_family_variance) == n_deltas)

        if (is.null(level_data) && is.null(level_formula)) {
            # Matrix is assumed to just have intercept terms
            delta_design_matrix <- matrix(1, nrow = n_levels, ncol = 1)
        } else {
            delta_design_matrix <- model.matrix(level_formula, model.frame(level_formula, level_data), ...)
        }

        if (length(dim(delta_family_mean)) == 2) {
            # Assumes that only the intercept term has been given
            dim(delta_family_mean) <- c(dim(delta_family_mean), 1)
        }

        delta <- list()
        # No delta's provided, so draw them from the family parameters
        for (level_index in 1 : n_levels) {
            # Just to get proportions correct
            level_delta <- delta_family_mean[, , 1]
            for (row in 1 : nrow(level_delta)) {
                for (column in 1 : ncol(level_delta)) {
                    level_delta[row, column] <- rnorm(
                        1,
                        sum(delta_family_mean[row, column, ] * delta_design_matrix[level_index, ]),
                        sqrt(delta_family_variance[row, column])
                    )
                }
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
        level_sample <- .logistic_generate(
            delta[[level_index]], panel_explanatory_variables[[level_index]],
            component_parameters, distributions, order
        )
        output_z <- c(output_z, level_sample$z)
        output_y <- c(output_y, level_sample$y)
    }
    output <- list(
        data = data.frame(z = output_z, y = output_y),
        explanatory_variables = explanatory_variables,
        delta = delta
    )
    if (length(output$delta) == 1) {
        output$delta <- delta[[1]]
    }
    output$distributions <- distributions
    output$component_parameters <- component_parameters
    output$order <- order

    return(output)
}
