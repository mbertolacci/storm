context('ptsm_hmm_generate')

test_that('ptsm_hmm_generate', {
    P <- matrix(
        c(
            0.7, 0.2, 0.1,
            0.4, 0.35, 0.25,
            0.2, 0.3, 0.5
        ),
        nrow=3,
        byrow=TRUE
    )

    n_samples <- 100
    data <- ptsm_hmm_generate(
        n_samples, P,
        c('gamma', 'gamma'),
        list(c(2, 1), c(0.5, 15)),
        z0=2
    )

    expect_equal(nrow(data), n_samples)
    expect_false(is.null(data$y))

    z_range <- range(data$z)
    expect_gte(z_range[1], 1)
    expect_lte(z_range[2], 3)
})
