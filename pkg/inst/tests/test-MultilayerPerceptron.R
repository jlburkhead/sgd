context("MultilayerPerceptron")

make_data <- function(p, k, binary = FALSE, n = 100)
    {
      X <- matrix(rnorm(n * p), n, p)
      if (binary) {
        if (k == 1)
          return(list(X = X, y = matrix(sample(0:1, n, replace = TRUE))))
        y <- sample(1:k, n, replace = TRUE)
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


test_that("MLPClassifier$Coef returns a list containing matricies with correct dimensions", {

    d <- make_data(10, 1, TRUE)
    mlp <- MLPClassifier(n_hidden = 10)
    mlp$Fit(d[["X"]], d[["y"]])
    coef <- mlp$Coef()

    expect_is(coef, "list")
    expect_is(coef[[1]], "matrix")
    expect_is(coef[[2]], "matrix")
    expect_equivalent(dim(coef[[1]]), c(11, 10))
    expect_equivalent(dim(coef[[2]]), c(11, 1))

    d <- make_data(5, 3, TRUE)
    mlp <- MLPClassifier(n_hidden = 10)
    mlp$Fit(d[["X"]], d[["y"]])
    coef <- mlp$Coef()

    expect_is(coef, "list")
    expect_is(coef[[1]], "matrix")
    expect_is(coef[[2]], "matrix")
    expect_equivalent(dim(coef[[1]]), c(6, 10))
    expect_equivalent(dim(coef[[2]]), c(11, 3))

})

test_that("l2_reg shrinks weights", {

    d <- make_data(10, 3, TRUE)
    p <- sapply(c(0, 1, 2, 5), function(reg) {
        set.seed(1)
        m <- MLPClassifier(epochs = 1e4, minibatch_size = 0, l2_reg = reg, n_hidden = 10)
        m$Fit(d[["X"]], d[["y"]])
        coefs <- m$Coef()
        coefs <- unlist(c(coefs[[1]][-1,], coefs[[2]][-1,]))
        sum(coefs ^ 2)
      })

    expect_true(all(diff(p) < 0))
      
})
