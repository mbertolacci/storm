context('alpha-conjugate')

check_result <- function(y) {
    expect_equal(length(y), 100)
    expect_false(any(y <= 0))
}

test_that('rgammashapeconjugate works when given parameters directly', {
    y <- rgammashapeconjugate(100, 1, 0, 1, 1)
    check_result(y)
})

test_that('rgammashapeconjugate works when given a little data', {
    x <- rgamma(10, 1.5)
    y <- rgammashapeconjugate(100, 1, x = x)
    check_result(y)
})

test_that('rgammashapeconjugate works when given a lot of data', {
    x <- rgamma(1000000, 1.5)
    y <- rgammashapeconjugate(100, 1, x = x)
    check_result(y)
})

test_that('rgammashapeconjugate works when given data and prior', {
    x <- rgamma(10, 1.5)
    y <- rgammashapeconjugate(100, 1, x = x, prior = c(0, 1, 1))
    check_result(y)
})
