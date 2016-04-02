# Estimate the parameters values based on their ergodic means
.mixtureEstimateAverages <- function(results) {
    return(colMeans(results$theta.sample))
}

# Estimate the parameters values based on the sample that had the highest posterior density
.mixtureEstimatePosteriorDensityMode <- function(results) {
    theta.mode <- results$theta.sample[which.max(results$log.posterior.density), ]
    return(theta.mode)
}
