context('independent_generate')

test_that('independent_generate', {
    data <- independent_generate(
        100,
        c(0.5, 0.2, 0.3),
        c('gamma', 'gamma'),
        list(c(2, 1), c(0.5, 15))
    )

    expect_equal(nrow(data), 100)
    expect_false(is.null(data$y))

    z_range <- range(data$z)
    expect_gte(z_range[1], 1)
    expect_lte(z_range[1], 3)
})
