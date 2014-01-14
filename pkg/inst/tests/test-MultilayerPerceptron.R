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


test_that("MultilayerPerceptron$Coef returns a list containing matricies with correct dimensions", {

    d <- make_data(10, 1, TRUE)
    mlp <- MultilayerPerceptron(n_hidden = 10)
    mlp$Fit(d[["X"]], d[["y"]])
    coef <- mlp$Coef()

    expect_is(coef, "list")
    expect_is(coef[[1]], "matrix")
    expect_is(coef[[2]], "matrix")
    expect_equivalent(dim(coef[[1]]), c(11, 10))
    expect_equivalent(dim(coef[[2]]), c(11, 1))

    d <- make_data(5, 3, TRUE)
    mlp <- MultilayerPerceptron(n_hidden = 10)
    mlp$Fit(d[["X"]], d[["y"]])
    coef <- mlp$Coef()

    expect_is(coef, "list")
    expect_is(coef[[1]], "matrix")
    expect_is(coef[[2]], "matrix")
    expect_equivalent(dim(coef[[1]]), c(6, 10))
    expect_equivalent(dim(coef[[2]]), c(11, 3))

})
