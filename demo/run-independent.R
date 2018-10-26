library(storm)

cat('\n--------- Gamma / Gamma\n')

n <- 5000
distributions <- c('gamma', 'gamma')
data <- independent_generate(
    n,
    c(0.5, 0.2, 0.3),
    distributions,
    list(c(2, 1), c(0.5, 15))
)

sampling_scheme <- list(
    # distributions = list(
    #     list(
    #         type = 'metropolis_hastings', use_mle = TRUE, use_observed_information = TRUE,
    #         observed_information_inflation_factor = 4
    #     ),
    #     list(
    #         type = 'metropolis_hastings', use_mle = TRUE, use_observed_information = TRUE,
    #         observed_information_inflation_factor = 0.025
    #     )
    # )
)

results <- independent_sample(
    n_samples = 10000, burn_in = 1000,
    y = data$y,
    distributions = distributions,
    thinning = list(distributions = 1, p = 1, z = 1, y_missing = 1),
    sampling_scheme = sampling_scheme,
    starting_values = 'bins',
    progress = TRUE
)

library(coda)

print(matrixStats::colQuantiles(results$sample$distribution[[1]], probs=c(0.025, 0.25, 0.5, 0.75, 0.975)))
print(matrixStats::colQuantiles(results$sample$distribution[[2]], probs=c(0.025, 0.25, 0.5, 0.75, 0.975)))
print(colMeans(results$sample$p))

print(rejectionRate(results$sample$distribution[[1]]))
print(rejectionRate(results$sample$distribution[[2]]))


cat('\n--------- Gengamma / Gengamma\n')

n <- 5000
distributions <- c('gengamma', 'gengamma')
data <- independent_generate(
    n,
    c(0.5, 0.2, 0.3),
    distributions,
    list(c(0.2, 0.7, 0.1), c(1.3, 1.5, 0.03))
)

sample <- independent_sample(
    n_samples=5000, burn_in=1000,
    y=c(data$y, rep(NA, 100)),
    distributions=distributions,
    thinning=list(distribution=1, z=1, y_missing=1),
    progress=TRUE
)

print(colMeans(sample$theta_sample))

cat('\n--------- Gamma / GEV\n')

n <- 5000
distributions <- c('gamma', 'gev')
data <- independent_generate(
    n,
    c(0.5, 0.2, 0.3),
    distributions,
    list(c(2, 0.7), c(5, 0.6, 0.5))
)

sample <- independent_sample(
    n_samples=5000, burn_in=1000,
    y=c(data$y, rep(NA, 100)),
    distributions=distributions,
    thinning=list(distribution=1, z=1, y_missing=1),
    progress=TRUE
)

print(colMeans(sample$theta_sample))

