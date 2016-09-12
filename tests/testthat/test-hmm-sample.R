context('ptsm_hmm_sample')

get_sample <- function(theta_sample_thinning = 0, z_sample_thinning = 0, y_missing_sample_thinning = 0) {
    distributions <- c('gamma', 'gamma')

    data <- ptsm_hmm_generate(
        1000,
        matrix(
            c(
                0.7, 0.2, 0.1,
                0.6, 0.3, 0.1,
                0.1, 0.4, 0.5
            ),
            nrow = 3,
            byrow = TRUE
        ),
        distributions,
        list(c(2, 1), c(0.5, 15))
    )

    ptsm_hmm_sample(
        n_samples = 200, burn_in = 100,
        y = c(data$y, rep(NA, 10)),
        distributions = distributions,
        theta_sample_thinning = theta_sample_thinning,
        z_sample_thinning = z_sample_thinning,
        y_missing_sample_thinning = y_missing_sample_thinning,
        verbose = 0
    )
}

test_that('theta_sample', {
    sample <- get_sample()
    expect_true(is.null(sample$theta_sample))

    sample <- get_sample(theta_sample_thinning = 1)
    expect_is(sample$theta_sample, 'mcmc')
    expect_equal(nrow(sample$theta_sample), 200)
    expect_true(all(
        colnames(sample$theta_sample) == c(
            'alpha[1]', 'beta[1]', 'alpha[2]', 'beta[2]',
            'p11', 'p12', 'p13',
            'p21', 'p22', 'p23',
            'p31', 'p32', 'p33'
        )
    ))
})

test_that('z_sample', {
    sample <- get_sample()
    expect_true(is.null(sample$z_sample))

    sample <- get_sample(z_sample_thinning = 10)
    expect_is(sample$z0_sample, 'mcmc')
    expect_is(sample$z_sample, 'mcmc')

    expect_equal(nrow(sample$z0_sample), 20)
    expect_equal(nrow(sample$z_sample), 20)

    expect_equal(ncol(sample$z0_sample), 1)
    expect_equal(ncol(sample$z_sample), 1010)
})

test_that('y_missing_sample', {
    sample <- get_sample()
    expect_true(is.null(sample$y_missing_sample))

    sample <- get_sample(y_missing_sample_thinning = 10)
    expect_is(sample$y_missing_sample, 'mcmc')
    expect_equal(nrow(sample$y_missing_sample), 20)
    expect_equal(ncol(sample$y_missing_sample), 10)
})
