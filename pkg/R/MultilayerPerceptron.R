#' Multilayer Perceptron Constructor
#'
#' Returns an instance of MultilayerPerceptron with supplied parameters or defaults.
#'
#' @param n_hidden number of hidden units (default = 10)
#' @param ... named arguments to pass to the constructor
#' @author Jake Burkhead <jlburkhead@@ucdavis.edu>
#' @export

MultilayerPerceptron <- function(n_hidden = 10, ...)
    {
        par <- .fill_params(...)
        return(new(.__C__Rcpp_.mlp, c(n_hidden = n_hidden, par)))
    }
