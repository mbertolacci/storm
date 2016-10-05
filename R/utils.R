# Given a vector or matrix, split into a list with one member per
# level in value_levels, container the elements/rows corresponding to the indices
# for that level in value.levels.
.levels_to_list <- function(values, value_levels) {
    if (length(dim(values)) > 2) {
        stop('Maximum dimensions exceeded')
    }
    if (class(values) == 'integer') {
        .levels_to_list_integer_vector(values, value_levels)
    } else {
        if (length(dim(values)) == 2) {
            .levels_to_list_numeric_matrix(values, value_levels)
        } else {
            .levels_to_list_numeric_vector(values, value_levels)
        }
    }
}

.levels_from_list_row <- function(values, value_levels) {
    n_dim <- length(dim(values[[1]]))
    if (n_dim > 2) {
        stop('Maximum dimensions exceeded')
    }

    n_levels <- nlevels(value_levels)
    if (n_dim == 2) {
        output <- matrix(0, nrow = length(value_levels), ncol = ncol(values[[1]]))
    } else {
        output <- rep(0, length(value_levels))
    }

    for (level_index in 1 : n_levels) {
        level_indices <- which(levels(value_levels)[level_index] == value_levels)
        if (n_dim == 2) {
            output[level_indices, ] <- values[[level_index]]
        } else {
            output[level_indices] <- values[[level_index]]
        }
    }

    return(output)
}

.levels_from_list_column <- function(values, value_levels) {
    n_levels <- nlevels(value_levels)
    output <- matrix(0, nrow = nrow(values[[1]]), ncol = length(value_levels))

    for (level_index in 1 : n_levels) {
        level_indices <- which(levels(value_levels)[level_index] == value_levels)
        output[, level_indices] <- values[[level_index]]
    }

    return(output)
}

# Roughly speaking, the inverse of the function above
.levels_from_list <- function(values, value_levels, dim = 0) {
    n_dim <- length(dim(values[[1]]))

    if (dim == 0 || n_dim == 1) {
        return(.levels_from_list_row(values, value_levels))
    } else {
        return(.levels_from_list_column(values, value_levels))
    }
}

.extend_list <- function(...) {
    lists <- list(...)
    output <- lists[[1]]
    for (value in lists[2 : length(lists)]) {
        for (name in names(value)) {
            output[[name]] <- value[[name]]
        }
    }
    return(output)
}
