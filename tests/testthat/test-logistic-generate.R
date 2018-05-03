context('logistic_generate')

test_that('logistic_generate generates non-panel data', {
    input_data <- data.frame(t = seq(0, 1, length.out = 100))

    formula <- ~ t + I(t ^ 2)
    distributions <- c('gamma', 'gamma')
    component_parameters <- list(c(1.5, 1), c(1.18, 10))
    delta <- rbind(
        c(1, 1, 1, 1, 1),
        c(-1, -1, -1, -1, -1)
    )

    output <- logistic_generate(
        input_data, formula, distributions, component_parameters,
        delta = delta
    )

    expect_equal(nrow(output$data), 100)
    expect_equal(output$delta, delta)
})

test_that('logistic_generate can generate panel data', {
    n_per_level <- 100
    n_levels <- 5
    input_data <- data.frame(
        t = rep(seq(0, 1, length.out = n_per_level), n_levels),
        group = factor(sapply(1 : n_levels, rep, n_per_level))
    )

    formula <- ~ t + I(t ^ 2)
    distributions <- c('gamma', 'gamma')
    component_parameters <- list(c(1.5, 1), c(1.18, 10))

    # Generate deltas from family parameters
    output <- logistic_generate(
        input_data, formula, distributions, component_parameters,
        delta_family_mean = matrix(0, nrow = 2, ncol = 5),
        delta_family_variance = matrix(1, nrow = 2, ncol = 5),
        panel_variable = 'group'
    )
    expect_equal(nrow(output$data), n_levels * n_per_level)
    expect_equal(length(output$delta), n_levels)
    expect_is(output$delta[[1]], 'matrix')

    # Provide deltas directly
    deltas <- list(
        matrix(1, nrow = 2, ncol = 5),
        matrix(2, nrow = 2, ncol = 5),
        matrix(3, nrow = 2, ncol = 5),
        matrix(4, nrow = 2, ncol = 5),
        matrix(5, nrow = 2, ncol = 5)
    )
    output <- logistic_generate(
        input_data, formula, distributions, component_parameters,
        delta = deltas, panel_variable = 'group'
    )
    expect_equal(output$delta, deltas)
})

test_that('logistic_generate can generate various orders', {
    input_data <- data.frame(t = seq(0, 1, length.out = 100))

    formula <- ~ t + I(t ^ 2)
    distributions <- c('gamma', 'gamma')
    component_parameters <- list(c(1.5, 1), c(1.18, 10))

    output <- logistic_generate(
        input_data, formula, distributions, component_parameters,
        delta = matrix(0.5, nrow = 2, ncol = 3),
        order = 0
    )
    expect_equal(nrow(output$data), 100)

    output <- logistic_generate(
        input_data, formula, distributions, component_parameters,
        delta = matrix(0.5, nrow = 2, ncol = 5),
        order = 1
    )
    expect_equal(nrow(output$data), 100)

    output <- logistic_generate(
        input_data, formula, distributions, component_parameters,
        delta = matrix(0.5, nrow = 2, ncol = 7),
        order = 2
    )
    expect_equal(nrow(output$data), 100)

    expect_error(
        logistic_generate(
            input_data, formula, distributions, component_parameters,
            delta = matrix(0.5, nrow = 2, ncol = 3),
            order = -1
        )
    )
})
