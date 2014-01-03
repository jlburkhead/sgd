SEED <- 1

context("shuffle_matrix")

test_that("shuffle_matrix permutes the rows of a matrix", {

    A <- matrix(runif(1000), 100)

    set.seed(SEED)
    
    shuffled_R <- A[sample(1:nrow(A), nrow(A)),]

    set.seed(SEED)
    expect_identical(shuffle_matrix(A), shuffled_R)

})
