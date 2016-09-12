context('gamma')

test_that('gamma_mle finds the MLE on a known sample', {
    expect_equal(
        gamma_mle(1 : 10),
        c(alpha = 2.728444, beta = 2.015801),
        tolerance = 1e-5
    )
})

test_that('gamma_mle raises an error when convergence is not reached', {
    expect_error(
        gamma_mle(1 : 10, absolute_tolerance = 1e-20),
        'gamma_mle did not converge within max_iterations'
    )
})
