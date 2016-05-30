context('base-sample')

test_that('.get_parameter_names', {
    expect_equal(
        .get_parameter_names(c('gamma', 'gamma')),
        c('alpha[1]', 'beta[1]', 'alpha[2]', 'beta[2]')
    )
    expect_equal(
        .get_parameter_names(c('gamma', 'gengamma')),
        c('alpha', 'beta', 'mu', 'sigma', 'Q')
    )
    expect_equal(
        .get_parameter_names(c('gamma', 'gev')),
        c('alpha', 'beta', 'mu', 'sigma', 'xi')
    )
})
