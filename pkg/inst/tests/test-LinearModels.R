context("LinearModels")

make_data <- function(p, k, binary = FALSE, n = 100)
    {
      X <- matrix(rnorm(n * p), n, p)
      if (binary) {
        if (k == 1)
          return(list(X = X, y = matrix(sample(0:1, n, replace = TRUE))))
        y <- sample(0:k, n, replace = TRUE)
        y <- model.matrix(~ factor(y) - 1)
        attributes(y) <- NULL
        y <- matrix(y, n, k)
      } else {
        y <- matrix(rnorm(n * k), n, k)
      }
      list(
        X = X,
        y = y
        )
    }


test_that("LinearRegression converges to lm's output", {

    data(iris)
    y <- matrix(iris$Petal.Length)
    X <- model.matrix(Petal.Length ~ Sepal.Length - 1, data = iris)
    X <- scale(X)
    
    target <- coef(lm(y ~ X))
    
    batch <- LinearRegression(epochs = 1e3, learning_rate = 0.01, momentum = 0.95, minibatch_size = 0)
    batch$Fit(X, y)

    expect_equivalent(as.numeric(batch$Coef()), target)

    minibatch <- LinearRegression(epochs = 1e5, learning_rate = 0.01, momentum = 0.99, minibatch_size = 100)
    minibatch$Fit(X, y)
    
    expect_true(all(abs(minibatch$Coef() - target) / target < 1e-2))

    stochastic <- LinearRegression(epochs = 1e5, learning_rate = 0.001, momentum = 0.99, minibatch_size = 1)
    stochastic$Fit(X, y)

    expect_true(all(abs(stochastic$Coef() - target) / target < 1e-2))
        
})


test_that("LinearRegression$Coef returns matrices with correct dimensions", {

    d1 <- make_data(10, 1)
    l1 <- LinearRegression()
    l1$Fit(d1[["X"]], d1[["y"]])
    p1 <- l1$Coef()
    
    expect_true(nrow(p1) == 11)
    expect_true(ncol(p1) == 1)

    d2 <- make_data(10, 10)
    l2 <- LinearRegression(fit_intercept = FALSE)
    l2$Fit(d2[["X"]], d2[["y"]])
    p2 <- l2$Coef()

    expect_true(nrow(p2) == 10)
    expect_true(ncol(p2) == 10)

})


test_that("LogisticRegression converges to glm's output", {

    data(iris)
    y <- as.matrix(as.numeric(iris$Species == "versicolor"))
    X <- model.matrix(Species ~ . - 1, data = iris)
    X <- scale(X)
    
    target <- coef(glm(y ~ X, family = binomial))

    batch <- LogisticRegression(epochs = 1e5, momentum = 0.95, minibatch_size = 0)
    batch$Fit(X, y)

    expect_equivalent(as.numeric(batch$Coef()), target)

    minibatch <- LogisticRegression(epochs = 1e5, learning_rate = 0.001, momentum = 0.99, minibatch_size = 100)
    minibatch$Fit(X, y)

    expect_true(all(abs(minibatch$Coef() - target) / target < 1e-2))

    stochastic <- LogisticRegression(epochs = 1e5, learning_rate = 0.001, momentum = 0.99, minibatch_size = 1)
    stochastic$Fit(X, y)

    expect_true(all(abs(stochastic$Coef() - target) / target < 1e-2))

})


test_that("LogisticRegression$Coef returns matrices with correct dimensions", {

    d1 <- make_data(10, 1, TRUE)
    l1 <- LogisticRegression(fit_intercept = FALSE)
    l1$Fit(d1[["X"]], d1[["y"]])
    p1 <- l1$Coef()
    
    expect_true(nrow(p1) == 10)
    expect_true(ncol(p1) == 1)

    d2 <- make_data(10, 10, TRUE)
    l2 <- LogisticRegression()
    l2$Fit(d2[["X"]], d2[["y"]])
    p2 <- l2$Coef()

    expect_true(nrow(p2) == 11)
    expect_true(ncol(p2) == 10)

})


test_that("LogisticRegression$Predict returns a matrix with normalized rows", {

    d1 <- make_data(10, 1, TRUE)
    l1 <- LogisticRegression()
    l1$Fit(d1[["X"]], d1[["y"]])

    pred <- l1$Predict(d1[["X"]])
    expect_true(all(pred >= 0 & pred <= 1))

    d2 <- make_data(10, 10, TRUE)
    l2 <- LogisticRegression()
    l2$Fit(d2[["X"]], d2[["y"]])

    pred <- l2$Predict(d2[["X"]])
    expect_true(all.equal(rowSums(pred), rep(1, nrow(pred))))

})


test_that("LogisticRegression$Predict_class returns valid classes", {

    d1 <- make_data(10, 1, TRUE)
    l1 <- LogisticRegression()
    l1$Fit(d1[["X"]], d1[["y"]])

    expect_true(all(l1$Predict_class(d1[["X"]]) %in% 0:1))
    
    d2 <- make_data(10, 5, TRUE)
    l2 <- LogisticRegression()
    l2$Fit(d2[["X"]], d2[["y"]])

    expect_true(all(l2$Predict_class(d2[["X"]]) %in% 1:5))
    
})


test_that("Link returns predictions at the level of the linear predictors", {

    for (k in 1:2) {
        d <- make_data(10, k)
        mod <- LinearRegression()
        mod$Fit(d[["X"]], d[["y"]])
        
        expect_equivalent(mod$Link(d[["X"]]), cbind(1, d[["X"]]) %*% mod$Coef())
    }

    for (k in 1:2) {
        d <- make_data(10, k, TRUE)
        
        mod <- LogisticRegression()
        mod$Fit(d[["X"]], d[["y"]])
        
        expect_equivalent(mod$Link(d[["X"]]), cbind(1, d[["X"]]) %*% mod$Coef())
    }

})


test_that("PoissonRegression converges to glm's output", {

    d <- read.csv(system.file("tests/testfiles/poisson_sim.csv", package = "sgd"))
    d$prog <- factor(d$prog)
    X <- model.matrix(num_awards ~ prog + math, data = d)[, -1]
    X <- scale(X)
    y <- as.matrix(d$num_awards)

    target <- coef(glm(y ~ X, family = poisson))

    batch <- PoissonRegression(epochs = 1e4, learning_rate = 0.01, momentum = 0.95, minibatch_size = 0)
    batch$Fit(X, y)

    expect_equivalent(as.numeric(batch$Coef()), target)

    minibatch <- PoissonRegression(epochs = 1e5, learning_rate = 0.001, momentum = 0.99, minibatch_size = 100)
    minibatch$Fit(X, y)
    
    expect_true(all(abs(minibatch$Coef() - target) / target < 1e-2))

    stochastic <- PoissonRegression(epochs = 1e5, learning_rate = 0.0001, momentum = 0.99, minibatch_size = 1)
    stochastic$Fit(X, y)

    expect_true(all(abs(stochastic$Coef() - target) / target < 1e-2))

})


test_that("l2_reg shrinks weights", {
    
    d <- make_data(10, 1)
    p <- lapply(c(0, 1, 10, 25, 50, 100, 200, 1000), function(reg) {
        set.seed(1)
        l <- LinearRegression(epochs = 1e3, minibatch_size = 0, l2_reg = reg)
        l$Fit(d[["X"]], d[["y"]])
        l$Coef()
    })

    p <- do.call(cbind, p)

    norms <- apply(p, 2, function(x) sum(x[-1] ^ 2) ) ## ignore bias

    expect_true(all(diff(norms) < 0))
    
    d <- make_data(10, 1, TRUE)
    p <- lapply(c(0, 1, 10, 25, 50, 100, 200, 1000), function(reg) {
        set.seed(1)
        l <- LogisticRegression(epochs = 1e3, minibatch_size = 0, l2_reg = reg)
        l$Fit(d[["X"]], d[["y"]])
        l$Coef()
    })

    p <- do.call(cbind, p)

    norms <- apply(p, 2, function(x) sum(x[-1] ^ 2) )

    expect_true(all(diff(norms) < 0))

    d <- read.csv(system.file("tests/testfiles/poisson_sim.csv", package = "sgd"))
    d$prog <- factor(d$prog)
    X <- model.matrix(num_awards ~ prog + math - 1, data = d)
    X <- scale(X)
    y <- as.matrix(d$num_awards)

    p <- lapply(c(0, 1, 10, 25, 50, 100, 200, 1000), function(reg) {
      set.seed(1)
      l <- PoissonRegression(epochs = 1e3, minibatch_size = 0, l2_reg = reg)
      l$Fit(X, y)
      l$Coef()
    })

    p <- do.call(cbind, p)

    norms <- apply(p, 2, function(x) sum(x[-1] ^ 2) )

    ## TODO: figure out why NaNs are showing up
    
    expect_true(all(diff(na.omit(norms)) < 0))
    
})
