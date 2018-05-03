context('independent_sample')

get_sample <- function(
    distributions = c('gamma', 'gamma'),
    parameters = list(c(2, 1), c(0.5, 15)),
    distributions_thinning = 0, p_thinning = 0, z_thinning = 0, y_missing_thinning = 0
) {
    distributions <- c('gamma', 'gamma')

    data <- independent_generate(
        1000,
        c(0.5, 0.2, 0.3),
        distributions,
        parameters
    )

    independent_sample(
        n_samples = 200, burn_in = 100,
        y = c(data$y, rep(NA, 10)),
        distributions = distributions,
        thinning=list(
            distributions = distributions_thinning,
            p = p_thinning,
            z = z_thinning,
            y_missing = y_missing_thinning
        )
    )$sample
}

test_that('sample$distribution has the right properties', {
    sample <- get_sample()
    expect_true(is.null(sample$distribution[[1]]))
    expect_true(is.null(sample$distribution[[2]]))

    sample <- get_sample(distributions_thinning = 1)
    expect_is(sample$distribution[[1]], 'mcmc')
    expect_equal(nrow(sample$distribution[[1]]), 200)
    expect_is(sample$distribution[[2]], 'mcmc')
    expect_equal(nrow(sample$distribution[[2]]), 200)
    expect_true(all(
        colnames(sample$distribution[[1]]) == c('alpha', 'beta')
    ))
    expect_true(all(
        colnames(sample$distribution[[2]]) == c('alpha', 'beta')
    ))
})

test_that('sample$p has the right properties', {
    sample <- get_sample()
    expect_true(is.null(sample$p))

    sample <- get_sample(p_thinning = 1)
    expect_is(sample$p, 'mcmc')
    expect_equal(nrow(sample$p), 200)
    expect_true(all(
        colnames(sample$p) == c('p1', 'p2', 'p3')
    ))
})

test_that('sample$z has the right properties', {
    sample <- get_sample()
    expect_true(is.null(sample$z))

    sample <- get_sample(z_thinning = 10)
    expect_is(sample$z, 'mcmc')
    expect_equal(nrow(sample$z), 20)
    expect_equal(ncol(sample$z), 1010)
})

test_that('sample$y_missing has the right properties', {
    sample <- get_sample()
    expect_true(is.null(sample$y_missing))

    sample <- get_sample(y_missing_thinning = 10)
    expect_is(sample$y_missing, 'mcmc')
    expect_equal(nrow(sample$y_missing), 20)
    expect_equal(ncol(sample$y_missing), 10)
})

test_that('various distributions work', {
    sample <- get_sample(
        distributions = c('gengamma', 'gengamma'),
        parameters = list(c(0.2, 0.7, 0.1), c(1.3, 1.5, 0.03))
    )
})
