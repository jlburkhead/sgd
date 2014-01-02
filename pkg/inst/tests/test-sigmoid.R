context("sigmoid")

test_that("sigmoid generates expected output", {

    sigmoid_R <- function(z) 1 / (1 + exp(-z)) ## should be the same as stats::plogis

    X <- seq(-20, 20, by = 0.01)
    random <- runif(100, -20, 20)
    
    ## test scalars
    for (x in X)
        expect_identical(sigmoid_R(x), sgd::sigmoid(x))

    for (x in random)
        expect_identical(sigmoid_R(x), sgd::sigmoid(x))

    ## test vectors
    expect_identical(sigmoid_R(X), sgd::sigmoid(X))
    expect_identical(sigmoid_R(random), sgd::sigmoid(random))

})
