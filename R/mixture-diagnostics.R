#' @import ggplot2
#' @importFrom reshape2 melt
#' @importFrom gridExtra grid.arrange
#' @importFrom scales hue_pal

# HACK(mike): the ggplot aes strings cause R package checks to fail, so pretend they are globals
utils::globalVariables(c('iteration', 'value', 'param', 'y', 'y.sample', '..density..'))

.printColumnsDiagnostics <- function(sample.values, columns=NA) {
    if (anyNA(columns)) {
        # If no columns were specified, use all of them
        columns <- 1 : dim(sample.values)[2]
    }

    data <- data.frame(sample.values[, columns])
    colnames(data) <- colnames(sample.values)[columns]
    data$iteration <- 1 : dim(sample.values)[1]

    melted.data <- melt(data, variable.name='param', id.var='iteration')
    print(
        ggplot(melted.data, aes(x=iteration, y=value, group=param, colour=param)) +
        geom_line() +
        geom_smooth(colour='black', se=FALSE, size=0.1, method='loess', span=0.1) +
        scale_colour_hue()
    )

    histograms <- lapply(1 : length(columns), function(i) {
        column <- columns[i]
        column.name <- colnames(sample.values)[column]
        colour <- hue_pal()(length(columns))[i]
        return(
            ggplot(data, aes_string(column.name, '..density..')) +
            geom_histogram(bins=100, colour=colour, fill=colour) +
            geom_density() +
            theme(legend.position="none")
        )
    })
    grid.arrange(grobs=histograms, ncol=2)

    acfs <- lapply(1 : length(columns), function(i) {
        column <- columns[i]
        bare.acf <- acf(sample.values[, column], plot = FALSE)
        data.acf <- with(bare.acf, data.frame(lag, acf))
        colour <- hue_pal()(length(columns))[i]
        return(
            ggplot(data.acf, aes(lag, acf)) +
            geom_bar(stat='identity', position='identity', fill=colour)
        )
    })
    grid.arrange(grobs=acfs, ncol=2)
}

.printSampleDensity <- function(y.original, y.sample) {
    original.data <- data.frame(y=y.original)
    y.sample <- y.sample[y.sample < max(y.original)]
    sample.data <- data.frame(y=y.sample)
    print(
        ggplot(original.data, aes(y, ..density..)) +
            geom_histogram(bins=100) +
            geom_histogram(bins=100, data=sample.data, aes(y.sample), fill=NA, colour='red')
    )
}

.printZPoints <- function(y, results) {
    # print(results$z.sample)
    z.data <- data.frame(y=y, z=colMeans(results$z.sample))
    print(
        ggplot(z.data, aes(y, z, colour=z)) + geom_point()
    )
}