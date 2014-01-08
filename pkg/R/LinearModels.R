#' Logistic Regression Constructor
#'
#' Returns an instance of LogisticRegression with supplied parameters or defaults.
#'
#' @param par named list of parameters to pass to the constructor
#' @return LogisticRegression object
#' @author Jake Burkhead <jlburkhead@@ucdavis.edu>
#' @export
LogisticRegression <- function(par = list(
                                   epochs = 100L,
                                   learning_rate = 0.01,
                                   momentum = 0.9,
                                   minibatch_size = 100L,
                                   l2_reg = 0,
                                   shuffle = TRUE,
                                   verbosity = 0L
                                   ))
    {
        return(new(logistic_regression, par))
    }
