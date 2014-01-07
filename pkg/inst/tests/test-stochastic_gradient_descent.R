context("stochastic_gradient_descent")

make_data <- function(p, k, n = 100)
    list(
        X = matrix(rnorm(n * p), n, p),
        y = matrix(sample(0:1, n * k, replace = TRUE), n, k)
        )

test_that("stochastic_gradient_descent converges to glm's output", {

    data(iris)
    y <- as.matrix(as.numeric(iris$Species == "versicolor"))
    X <- model.matrix(Species ~ ., data = iris)
    X[,-1] <- scale(X[,-1])
    
    target <- coef(glm(y ~ X - 1, family = binomial))
    
    params_batch <- stochastic_gradient_descent(
        X,
        y,
        1e4,
        0.01,
        0.95,
        nrow(X)
        )
    
    params_minibatch <- stochastic_gradient_descent(
        X,
        y,    
        1e5,
        0.001,
        0.99,
        100
        )
    
    params_stochastic <- stochastic_gradient_descent(
        X,
        y,    
        1e5,
        0.001,
        0.99,
        1
        )
        
    expect_equivalent(as.numeric(params_batch), target)
    expect_true(
        all.equal(
            target,
            as.numeric(params_minibatch),
            tolerance = 1e-4,
            check.attributes = FALSE
            ))
    expect_true(
        all.equal(
            target,
            as.numeric(params_stochastic),
            tolerance = 1e-3,
            check.attributes = FALSE
            ))

})


test_that("stochastic_gradient_descent returns matrices with correct dimensions", {

    d1 <- make_data(10, 1)
    p1 <- stochastic_gradient_descent(d1[["X"]], d1[["y"]], 10, 0.1, 0.9)

    expect_true(nrow(p1) == 10)
    expect_true(ncol(p1) == 1)

    d2 <- make_data(10, 10)
    p2 <- stochastic_gradient_descent(d2[["X"]], d2[["y"]], 10, 0.1, 0.9)

    expect_true(nrow(p2) == 10)
    expect_true(ncol(p2) == 10)

})


test_that("l2_reg shrinks weights", {

    d <- make_data(10, 1)
    p <- lapply(c(0, 1, 10, 25, 50, 100, 200, 1000), function(reg) {
        set.seed(1)
        stochastic_gradient_descent(d[["X"]], d[["y"]], 1000, 0.01, 0.9, l2_reg = reg)
    })

    p <- do.call(cbind, p)

    norms <- apply(p, 2, function(x) sum(x ^ 2) )

    expect_true(all(diff(norms) < 0))

})

    
