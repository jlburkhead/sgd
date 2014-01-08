.default_logistic_params <- list(
  epochs = 100L,
  learning_rate = 0.01,
  momentum = 0.9,
  minibatch_size = 100L,
  l2_reg = 0,
  shuffle = TRUE,
  verbosity = 0L
  )

#' Logistic Regression Constructor
#'
#' Returns an instance of LogisticRegression with supplied parameters or defaults.
#'
#' @param ... named parameters to pass to the constructor
#' @return LogisticRegression object
#' @author Jake Burkhead <jlburkhead@@ucdavis.edu>
#' @export
LogisticRegression <- function(...)
    {
        par <- list(...)
        drop <- setdiff(names(par), names(.default_logistic_params))
        if (length(drop)) {
          msg <- paste("Ignoring the following parameters:", paste(drop, collapse = ", "))
          warning(msg)
          par <- par[!names(par) %in% drop]
        }
        add <- setdiff(names(.default_logistic_params), names(par))
        par <- c(par, .default_logistic_params[add])
        return(new(logistic_regression, par))
    }
