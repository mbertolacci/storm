context('logistic_sample')

DEFAULT_DISTRIBUTIONS <- c('gamma', 'gamma')
DEFAULT_SAMPLING_SCHEME <- list(
    distributions = .default_sampling_scheme(DEFAULT_DISTRIBUTIONS)
)

test_that('.get_logistic_sample_prior sets distributions', {
    # Should set a default (tested elsewhere)
    prior <- .get_logistic_sample_prior(NULL, DEFAULT_DISTRIBUTIONS, 1, 2, 1, DEFAULT_SAMPLING_SCHEME)
    expect_true(!is.null(prior$distributions))

    # Should return what is passed through
    prior_distributions <- list(
        list(type = 'uniform', bounds = matrix(c(0, 0, 500, 500), nrow = 2)),
        list(type = 'uniform', bounds = matrix(c(0, 0, 500, 500), nrow = 2))
    )
    prior <- .get_logistic_sample_prior(list(distributions = prior_distributions), DEFAULT_DISTRIBUTIONS, 1, 2, 1)
    expect_equal(prior$distributions, prior_distributions)
})

test_that('.get_logistic_sample_prior uses normal prior when there is only one level', {
    prior <- .get_logistic_sample_prior(NULL, DEFAULT_DISTRIBUTIONS, 1, 2, 1, DEFAULT_SAMPLING_SCHEME)

    # One level defaults to a normal prior
    expect_equal(prior$logistic$type, 'normal')
    expect_false(is.null(prior$logistic$mean))
    expect_false(is.null(prior$logistic$variance))
})

test_that('.get_logistic_sample_prior uses hierarchical prior when there is more than one level', {
    prior <- .get_logistic_sample_prior(NULL, DEFAULT_DISTRIBUTIONS, 2, 2, 1, DEFAULT_SAMPLING_SCHEME)

    # >1 level defaults to a hierarchical prior
    expect_equal(prior$logistic$type, 'hierarchical')
    expect_false(prior$logistic$is_gp)
    expect_false(is.null(prior$logistic$mean))
    expect_false(is.null(prior$logistic$variance))
})

test_that('.get_logistic_sample_prior with gaussian process prior', {
    base_prior <- list(logistic = list(type = 'hierarchical', is_gp = TRUE))

    prior <- .get_logistic_sample_prior(base_prior, DEFAULT_DISTRIBUTIONS, 9, 2, 2, DEFAULT_SAMPLING_SCHEME)
    expect_true(prior$logistic$is_gp)
    expect_false(is.null(prior$logistic$tau_squared))
    # There are now effectively the original 2 plus 9 new level variables
    expect_equal(dim(prior$logistic$mean$mean)[[3]], 11)

    # If there are fewer than 10 levels, use all of the basis vectors
    expect_equal(prior$logistic$n_gp_bases, 9)

    prior <- .get_logistic_sample_prior(base_prior, DEFAULT_DISTRIBUTIONS, 20, 2, 2, DEFAULT_SAMPLING_SCHEME)
    # If there are more than 10, default to max(10, n_levels / 10)
    expect_equal(prior$logistic$n_gp_bases, 10)
    prior <- .get_logistic_sample_prior(base_prior, DEFAULT_DISTRIBUTIONS, 200, 2, 2, DEFAULT_SAMPLING_SCHEME)
    expect_equal(prior$logistic$n_gp_bases, 20)
})

test_that('logistic_sample should reject panel data with mismatched levels', {
    data <- data.frame(
        group = as.factor(1 : 10)
    )

    # These levels are mismatched because the ordering for level_data is '1', '10', '2', etc, while for input_data it
    # is simply 1 : 110
    level_data <- data.frame(
        group = as.factor(as.character(1 : 10))
    )
    expect_error({
        logistic_sample(1, 1, data, ~ ., level_data = level_data, panel_variable = 'group')
    }, 'Levels in data and level_data must be equal')

    # Duplicated level in level_data
    level_data <- data.frame(
        group = as.factor(c(1 : 10, 10))
    )
    expect_error({
        logistic_sample(1, 1, data, ~ ., level_data = level_data, panel_variable = 'group')
    }, 'Number of rows in level_data should equal number of levels')

    # NOTE: The success case is implicitly handled in other tests
})

test_that('logistic_sample on non-panel data', {
    input_data <- data.frame(t = seq(0, 1, length.out = 100))

    formula <- ~ t + I(t ^ 2)
    distributions <- c('gamma', 'gamma')
    component_parameters <- list(c(1.5, 1), c(1.18, 10))
    n_deltas <- 5
    delta <- rbind(
        c(1, 1, 1, 1, 1),
        c(-1, -1, -1, -1, -1)
    )

    generated <- logistic_generate(
        input_data, formula, distributions, component_parameters,
        delta = delta
    )
    data <- input_data
    data$y <- generated$data$y

    n_samples <- 20

    output <- logistic_sample(
        n_samples, 5,
        data, y ~ t + I(t ^ 2), distributions
    )

    expect_equal(dim(output$sample$distribution[[1]]), c(n_samples, 2))
    expect_equal(dim(output$sample$distribution[[2]]), c(n_samples, 2))
    expect_equal(dim(output$sample$delta), c(n_samples, 2, n_deltas))
})

# get_sample <- function(n_levels, n_per_level, n_samples) {
#     input_data <- data.frame(
#         t=rep(seq(0, 1, length.out=n_per_level), n_levels),
#         group=factor(sapply(1 : n_levels, rep, n_per_level))
#     )

#     formula <- ~ t + I(t ^ 2)
#     distributions <- c('gamma', 'gamma')
#     component_parameters <- list(c(1.5, 1), c(1.18, 10))

#     n_deltas <- 5

#     # Generate deltas from family parameters
#     generated <- logistic_generate(
#         input_data, formula, distributions, component_parameters,
#         delta_family_mean=matrix(0, nrow=2, ncol=n_deltas),
#         delta_family_variance=matrix(1, nrow=2, ncol=n_deltas),
#         panel_variable='group'
#     )

#     data <- input_data
#     data$y <- generated$data$y

#     logistic_sample(
#         n_samples=n_samples, burn_in=5,
#         data, y ~ t + I(t ^ 2), distributions,
#         theta_sample_thinning=1, z_sample_thinning=1, y_missing_sample_thinning=0,
#         panel_variable='group',
#         verbose=0
#     )
# }

# test_that('logistic_sample on panel data', {
#     n_levels <- 2
#     n_per_level <- 100
#     n_samples <- 20
#     n_deltas <- 5
#     output <- get_sample(n_levels, n_per_level, n_samples)

#     # Check column names match (yuck)
#     expect_equal(unname(colnames(output$theta_sample)), c(
#         'alpha[1]', 'beta[1]', 'alpha[2]', 'beta[2]',
#         '(Intercept):1[1]', 't:1[1]', 'I(t^2):1[1]', 'z2(t-1):1[1]', 'z3(t-1):1[1]',
#         '(Intercept):1[2]', 't:1[2]', 'I(t^2):1[2]', 'z2(t-1):1[2]', 'z3(t-1):1[2]',
#         '(Intercept):2[1]', 't:2[1]', 'I(t^2):2[1]', 'z2(t-1):2[1]', 'z3(t-1):2[1]',
#         '(Intercept):2[2]', 't:2[2]', 'I(t^2):2[2]', 'z2(t-1):2[2]', 'z3(t-1):2[2]',
#         '(Intercept):family_mean[1]', 't:family_mean[1]', 'I(t^2):family_mean[1]', 'z2(t-1):family_mean[1]',
#             'z3(t-1):family_mean[1]',
#         '(Intercept):family_mean[2]', 't:family_mean[2]', 'I(t^2):family_mean[2]', 'z2(t-1):family_mean[2]',
#             'z3(t-1):family_mean[2]',
#         '(Intercept):family_variance[1]', 't:family_variance[1]', 'I(t^2):family_variance[1]',
#             'z2(t-1):family_variance[1]', 'z3(t-1):family_variance[1]',
#         '(Intercept):family_variance[2]', 't:family_variance[2]', 'I(t^2):family_variance[2]',
#             'z2(t-1):family_variance[2]', 'z3(t-1):family_variance[2]'
#     ))
#     expect_equal(nrow(output$theta_sample), n_samples)
#     expect_equal(ncol(output$theta_sample), 4 + 2 * n_deltas * (n_levels + 2))
#     expect_equal(nrow(output$z_sample), n_samples)
#     expect_equal(ncol(output$z_sample), n_levels * n_per_level)
# })

# test_that('logistic_sample_y', {
#     n_levels <- 2
#     n_per_level <- 100
#     n_samples <- 10

#     output <- get_sample(n_levels, n_per_level, n_samples)
#     y_sample_output <- logistic_sample_y(output)

#     expect_equal(nrow(y_sample_output$y_sample), n_samples)
#     expect_equal(nrow(y_sample_output$y_sample_z), n_samples)
# })
