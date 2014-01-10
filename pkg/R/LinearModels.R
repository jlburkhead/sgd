.default_params <- list(
  epochs = 100L,
  learning_rate = 0.01,
  momentum = 0.9,
  minibatch_size = 100L,
  l2_reg = 0,
  shuffle = TRUE
  )

.fill_params <- function(...)
    {
        par <- list(...)
        drop <- setdiff(names(par), names(.default_params))
        if (length(drop)) {
          msg <- paste("Ignoring the following parameters:", paste(drop, collapse = ", "))
          warning(msg)
          par <- par[!names(par) %in% drop]
        }
        add <- setdiff(names(.default_params), names(par))
        par <- c(par, .default_params[add])
        
        return(par)
    }


#' Logistic Regression Constructor
#'
#' Returns an instance of LogisticRegression with supplied parameters or defaults.
#'
#' @param ... named arguments to pass to the constructor
#' @return LogisticRegression object
#' @author Jake Burkhead <jlburkhead@@ucdavis.edu>
#' @export

LogisticRegression <- function(...)
    {
        par <- .fill_params(...)
        return(new(.__C__Rcpp_.logistic_regression, par))
    }

#' Linear Regression Constructor
#'
#' Returns an instance of LinearRegression with supplied parameters or defaults
#'
#' @param ... named arguments to pass to the constructor
#' @return LinearRegression object
#' @author Jake Burkhead <jlburkhead@@ucdavis.edu>
#' @export

LinearRegression <- function(...)
    {
        par <- .fill_params(...)
        return(new(.__C__Rcpp_.linear_regression, par))
    }
      

#' Poisson Regression Constructor
#'
#' Returns an instance of PoissonRegression with supplied parameters or defaults
#'
#' @param ... named arguments to pass to the constructor
#' @return PoissonRegression object
#' @author Jake Burkhead <jlburkhead@@ucdavis.edu>
#' @export

PoissonRegression <- function(...)
  {
      par <- .fill_params(...)
      return(new(.__C__Rcpp_.poisson_regression, par))
  }
    
