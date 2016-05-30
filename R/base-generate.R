.y_given_z <- function(z, distributions, component_parameters) {
    y <- rep(0, length(z))
    y[z == 1] <- 0

    for (index in 1 : length(distributions)) {
        component_z <- index + 1

        parameters <- component_parameters[[index]]

        if (distributions[index] == 'gamma') {
            y[z == component_z] <- rgamma(
                length(which(z == component_z)), parameters[1], scale=parameters[2]
            )
        } else if (distributions[index] == 'gev') {
            y[z == component_z] <- rgev(
                length(which(z == component_z)), parameters[1], parameters[2], parameters[3]
            )
        } else if (distributions[index] == 'gengamma') {
            y[z == component_z] <- rgengamma(
                length(which(z == component_z)), parameters[1], parameters[2], parameters[3]
            )
        }
    }

    return(y)
}
