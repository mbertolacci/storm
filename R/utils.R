#' @useDynLib positivemixtures

# Given a vector or matrix, split into a list with one member per
# level in value.levels, container the elements/rows corresponding to the indices
# for that level in value.levels.
.levels_to_list <- function(values, value.levels) {
    if (length(dim(values)) > 2) {
        stop('Maximum dimensions exceeded')
    }
    lapply(levels(value.levels), function(level) {
        if (length(dim(values)) == 2) {
            values[value.levels == level, ]
        } else {
            values[value.levels == level]
        }
    })
}

# Roughly speaking, the inverse of the function above
.levels_from_list <- function(values, value.levels) {
    n.dim <- length(dim(values[[1]]))
    if (n.dim > 2) {
        stop('Maximum dimensions exceeded')
    }

    n.levels <- nlevels(value.levels)
    if (n.dim == 2) {
        output <- matrix(0, nrow=length(value.levels), ncol=ncol(values[[1]]))
    } else {
        output <- rep(0, length(value.levels))
    }

    for (level.index in 1 : n.levels) {
        level.indices <- which(levels(value.levels)[level.index] == value.levels)
        if (n.dim == 2) {
            output[level.indices, ] <- values[[level.index]]
        } else {
            output[level.indices] <- values[[level.index]]
        }
    }

    return(output)
}
