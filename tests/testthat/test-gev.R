context('gev')

test_that('dgev', {
    # Q = 0
    expect_equal(dgev(0, 0, 1, 0), 0.3678794, tolerance = 1e-6)
    expect_equal(dgev(1, 0, 1, 0), 0.2546464, tolerance = 1e-6)
    # Q != 0
    expect_equal(dgev(0, 0, 1, 0.5), 0.3678794, tolerance = 1e-6)
    expect_equal(dgev(0, 0, 1, -0.5), 0.3678794, tolerance = 1e-6)
    # mu != 0
    expect_equal(dgev(2, 1, 1, 0.5), 0.1899794, tolerance = 1e-6)
})
