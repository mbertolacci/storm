devtools::load_all('positivemixtures')

cat('\n--------- Gamma / Gamma\n')

n <- 5000
P <- matrix(
    c(
        0.7, 0.2, 0.1,
        0.6, 0.3, 0.1,
        0.1, 0.4, 0.5
    ),
    nrow=3,
    byrow=TRUE
)

distributions <- c('gamma', 'gamma')
data <- ptsm_hmm_generate(
    n, P,
    distributions,
    list(c(2, 1), c(0.5, 15))
)

sample <- ptsm_hmm_sample(
    n_samples=5000, burn_in=1000,
    y=c(data$y, rep(NA, 100)),
    distributions=distributions,
    theta_sample_thinning=1, z_sample_thinning=1, y_missing_sample_thinning=1,
    verbose=0
)
print(colMeans(sample$theta_sample))

cat('\n--------- Gengamma / Gengamma\n')

n <- 5000
distributions <- c('gengamma', 'gengamma')
data <- ptsm_hmm_generate(
    n,
    P,
    distributions,
    list(c(0.2, 0.7, 0.1), c(1.3, 1.5, 0.03))
)

sample <- ptsm_hmm_sample(
    n_samples=5000, burn_in=1000,
    y=c(data$y, rep(NA, 100)),
    distributions=distributions,
    theta_sample_thinning=1, z_sample_thinning=1, y_missing_sample_thinning=1,
    verbose=0
)

print(colMeans(sample$theta_sample))

cat('\n--------- Gamma / GEV\n')

n <- 5000
distributions <- c('gamma', 'gev')
data <- ptsm_hmm_generate(
    n,
    P,
    distributions,
    list(c(2, 0.7), c(5, 0.6, 0.5))
)

sample <- ptsm_hmm_sample(
    n_samples=5000, burn_in=1000,
    y=c(data$y, rep(NA, 100)),
    distributions=distributions,
    theta_sample_thinning=1, z_sample_thinning=1, y_missing_sample_thinning=1,
    verbose=0
)

print(colMeans(sample$theta_sample))

