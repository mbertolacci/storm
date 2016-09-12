context('gengamma')

test_that('dgengamma', {
    # Q = 0
    expect_equal(dgengamma(0, 0, 1, 0), 0)
    expect_equal(dgengamma(1, 0, 1, 0), 0.3989423, tolerance = 1e-6)
    # Q != 0
    expect_equal(dgengamma(1, 0, 1, 0.5), 0.3907336, tolerance = 1e-6)
    expect_equal(dgengamma(1, 0, 1, -0.5), 0.3907336, tolerance = 1e-6)
    # mu != 0
    expect_equal(dgengamma(2, 1, 1, 0.5), 0.1868148, tolerance = 1e-6)
    # Log density option
    expect_equal(dgengamma(1, 0, 1, 0, log = TRUE), log(0.3989423), tolerance = 1e-6)
})

test_that('rgengamma', {
    # Q = 0
    expect_gt(rgengamma(1, 0, 1, 0), 0)
    # Q != 0
    expect_gt(rgengamma(1, 0, 1, 0.5), 0)
})

test_that('gengamma_mle', {
    mle_result <- gengamma_mle(c(
        2.2947923, 1.0779932, 0.1350498, 0.4083266, 0.1463441, 2.5861788,
        1.2465664, 1.7308418, 0.3341165, 0.7000765, 0.2828153, 1.2720446,
        0.5713153, 0.1933244, 0.2145998, 3.3993156, 0.6608248, 0.7635703,
        0.1061541, 0.2569416
    ))

    # Here's one I prepared earlier
    expect_equal(
        mle_result$par,
        c(mu = -0.6733181, sigma = 1.0132464, Q = -0.1953667),
        tolerance = 1e-7
    )
})

test_that('gengamma_llgradient', {
    grad <- gengamma_llgradient(1 : 10, 0, 1, 0.1)
    expect_equal(grad, c(16.58060, 20.50885, -10.01780), tolerance = 1e-6)
})

test_that('gengamma_llhessian', {
    hessian <- gengamma_llhessian(1 : 10, 0, 1, 0.1)
    expected.hessian <- rbind(
        c(-11.65806, -34.73590,  15.74698),
        c(-34.73590, -84.58268,  30.56126),
        c( 15.74698,  30.56126, -11.74514)
    )
    expect_equal(hessian, expected.hessian, tolerance = 1e-6)
})
