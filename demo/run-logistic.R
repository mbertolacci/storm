library(coda)
library(storm)

# cat('\n--------- Gamma / Gamma (single, 0-order)\n')

# distributions <- c('gamma', 'gamma')
# input_data <- data.frame(t=seq(0, 1, length.out=2500))

# output <- logistic_generate(
#     input_data,
#     ~ t, order=0,
#     distributions,
#     component_parameters=list(c(1.3, 1), c(1.18, 11)),
#     delta=rbind(
#         c(1, 0.5),
#         c(1, -0.5)
#     )
# )

# data <- input_data
# data$y <- output$data$y

# results <- logistic_sample(
#     n_samples=1000, burn_in=1000,
#     data, y ~ t, order=0,
#     distributions=distributions,
#     theta_sample_thinning=1, z_sample_thinning=0, y_missing_sample_thinning=0,
#     verbose=0
# )
# print(colMeans(results$sample$distribution[[1]]))
# print(colMeans(results$sample$distribution[[2]]))

# cat('\n--------- Gamma / Gamma (single, 1-order)\n')

set.seed(100)

# distributions <- c('gamma', 'gamma')
# input_data <- data.frame(t=seq(0, 1, length.out=1000))

# output <- logistic_generate(
#     input_data,
#     ~ t, order=1,
#     distributions,
#     component_parameters=list(c(1.3, 1), c(1.18, 11)),
#     delta=rbind(
#         c(1, 1, 0.5, 0),
#         c(1, -1, 0, 0.5)
#     )
# )

# data <- input_data
# data$y <- output$data$y

# results <- logistic_sample(
#     n_samples=1000, burn_in=1000,
#     data, y ~ t, order=1,
#     distributions=distributions,
#     verbose=0
# )
# print(colMeans(results$sample$distribution[[1]]))
# print(colMeans(results$sample$distribution[[2]]))
# print(apply(results$sample$delta, 2 : 3, mean))

cat('\n--------- Gamma / Gamma (panel)\n')

n_per_level <- 1000
n_levels <- 5
distributions <- c('gamma', 'gamma')
input_data <- data.frame(
    t=rep(seq(0, 1, length.out=n_per_level), n_levels),
    group=factor(sapply(1 : n_levels, rep, n_per_level))
)

output <- logistic_generate(
    input_data,
    ~ t,
    distributions,
    component_parameters=list(c(1.3, 1), c(1.18, 11)),
    delta_family_mean=matrix(0, nrow=2, ncol=4),
    delta_family_variance=matrix(1, nrow=2, ncol=4),
    panel_variable='group'
)

data <- input_data
data$y <- output$data$y

results <- logistic_sample(
    n_samples=10000, burn_in=1000,
    data, y ~ t,
    distributions=distributions,
    panel_variable='group'
)
print(colMeans(results$sample$distribution[[1]]))
print(colMeans(results$sample$distribution[[2]]))
# print(apply(results$sample$delta, 2 : 4, mean))
print(apply(results$sample$delta_family_mean, 2 : 4, mean))

# hpd <- HPDinterval(sample$theta_sample)
# for (i in 1 : (n_levels + 2)) {
#     print(hpd[(5 + (i - 1) * 9) : (9 + (i - 1) * 9), ])
#     print(hpd[(11 + (i - 1) * 9) : (15 + (i - 1) * 9), ])
#     # print(colMeans(sample$theta_sample)[(5 + (i - 1) * 8) : (8 + (i - 1) * 8)])
#     # print(colMeans(sample$theta_sample)[(9 + (i - 1) * 8) : (12 + (i - 1) * 8)])
# }

# cat('\n--------- Gengamma / Gengamma\n')

# n <- 5000
# distributions <- c('gengamma', 'gengamma')
# data <- independent_generate(
#     n,
#     c(0.5, 0.2, 0.3),
#     distributions,
#     list(c(0.2, 0.7, 0.1), c(1.3, 1.5, 0.03))
# )

# sample <- independent_sample(
#     n_samples=5000, burn_in=1000,
#     y=c(data$y, rep(NA, 100)),
#     distributions=distributions,
#     theta_sample_thinning=1, z_sample_thinning=1, y_missing_sample_thinning=1,
#     verbose=0
# )

# print(colMeans(sample$theta_sample))

# cat('\n--------- Gamma / GEV\n')

# n <- 5000
# distributions <- c('gamma', 'gev')
# data <- independent_generate(
#     n,
#     c(0.5, 0.2, 0.3),
#     distributions,
#     list(c(2, 0.7), c(5, 0.6, 0.5))
# )

# sample <- independent_sample(
#     n_samples=5000, burn_in=1000,
#     y=c(data$y, rep(NA, 100)),
#     distributions=distributions,
#     theta_sample_thinning=1, z_sample_thinning=1, y_missing_sample_thinning=1,
#     verbose=0
# )

# print(colMeans(sample$theta_sample))

