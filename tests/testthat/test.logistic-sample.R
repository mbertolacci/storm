context('ptsm_logistic_sample')

test_that('ptsm_logistic_sample on non-panel data', {
    input_data <- data.frame(t=seq(0, 1, length.out=100))

    formula <- ~ t + I(t ^ 2)
    distributions <- c('gamma', 'gamma')
    component_parameters <- list(c(1.5, 1), c(1.18, 10))
    n_deltas <- 5
    delta <- rbind(
        c(1, 1, 1, 1, 1),
        c(-1, -1, -1, -1, -1)
    )

    generated <- ptsm_logistic_generate(
        input_data, formula, distributions, component_parameters,
        delta=delta
    )
    data <- input_data
    data$y <- generated$data$y

    n_samples <- 20

    output <- ptsm_logistic_sample(
        n_samples=n_samples, burn_in=5,
        data, y ~ t + I(t ^ 2), distributions,
        theta_sample_thinning=1, z_sample_thinning=1, y_missing_sample_thinning=0,
        verbose=0
    )

    expect_equal(nrow(output$theta_sample), n_samples)
    expect_equal(ncol(output$theta_sample), 4 + 2 * n_deltas)
    expect_equal(nrow(output$z_sample), n_samples)
    expect_equal(ncol(output$z_sample), 100)
})

get_sample <- function(n_levels, n_per_level, n_samples) {
    input_data <- data.frame(
        t=rep(seq(0, 1, length.out=n_per_level), n_levels),
        group=factor(sapply(1 : n_levels, rep, n_per_level))
    )

    formula <- ~ t + I(t ^ 2)
    distributions <- c('gamma', 'gamma')
    component_parameters <- list(c(1.5, 1), c(1.18, 10))

    n_deltas <- 5

    # Generate deltas from family parameters
    generated <- ptsm_logistic_generate(
        input_data, formula, distributions, component_parameters,
        delta_family_mean=matrix(0, nrow=2, ncol=n_deltas),
        delta_family_variance=matrix(1, nrow=2, ncol=n_deltas),
        panel_variable='group'
    )

    data <- input_data
    data$y <- generated$data$y

    ptsm_logistic_sample(
        n_samples=n_samples, burn_in=5,
        data, y ~ t + I(t ^ 2), distributions,
        theta_sample_thinning=1, z_sample_thinning=1, y_missing_sample_thinning=0,
        panel_variable='group',
        verbose=0
    )
}

test_that('ptsm_logistic_sample on panel data', {
    n_levels <- 2
    n_per_level <- 100
    n_samples <- 20
    n_deltas <- 5
    output <- get_sample(n_levels, n_per_level, n_samples)

    # Check column names match (yuck)
    expect_equal(unname(colnames(output$theta_sample)), c(
        'alpha[1]', 'beta[1]', 'alpha[2]', 'beta[2]',
        '(Intercept):1[1]', 't:1[1]', 'I(t^2):1[1]', 'z2(t-1):1[1]', 'z3(t-1):1[1]',
        '(Intercept):1[2]', 't:1[2]', 'I(t^2):1[2]', 'z2(t-1):1[2]', 'z3(t-1):1[2]',
        '(Intercept):2[1]', 't:2[1]', 'I(t^2):2[1]', 'z2(t-1):2[1]', 'z3(t-1):2[1]',
        '(Intercept):2[2]', 't:2[2]', 'I(t^2):2[2]', 'z2(t-1):2[2]', 'z3(t-1):2[2]',
        '(Intercept):family_mean[1]', 't:family_mean[1]', 'I(t^2):family_mean[1]', 'z2(t-1):family_mean[1]',
            'z3(t-1):family_mean[1]',
        '(Intercept):family_mean[2]', 't:family_mean[2]', 'I(t^2):family_mean[2]', 'z2(t-1):family_mean[2]',
            'z3(t-1):family_mean[2]',
        '(Intercept):family_variance[1]', 't:family_variance[1]', 'I(t^2):family_variance[1]',
            'z2(t-1):family_variance[1]', 'z3(t-1):family_variance[1]',
        '(Intercept):family_variance[2]', 't:family_variance[2]', 'I(t^2):family_variance[2]',
            'z2(t-1):family_variance[2]', 'z3(t-1):family_variance[2]'
    ))
    expect_equal(nrow(output$theta_sample), n_samples)
    expect_equal(ncol(output$theta_sample), 4 + 2 * n_deltas * (n_levels + 2))
    expect_equal(nrow(output$z_sample), n_samples)
    expect_equal(ncol(output$z_sample), n_levels * n_per_level)
})

test_that('ptsm_logistic_sample_y', {
    n_levels <- 2
    n_per_level <- 100
    n_samples <- 10

    output <- get_sample(n_levels, n_per_level, n_samples)
    y_sample_output <- ptsm_logistic_sample_y(output)

    expect_equal(nrow(y_sample_output$y_sample), n_samples)
    expect_equal(nrow(y_sample_output$y_sample_z), n_samples)
})
