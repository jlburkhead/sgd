#include <RcppArmadillo.h>

#include "stochastic_gradient_descent.hpp"

using namespace Rcpp;

class logistic_regression {
public:
  logistic_regression(List l) : epochs(as<int>(l["epochs"])),
				learning_rate(as<double>(l["learning_rate"])),
				momentum(as<double>(l["momentum"])),
				minibatch_size(as<int>(l["minibatch_size"])),
				l2_reg(as<double>(l["l2_reg"])),
				shuffle(as<bool>(l["shuffle"])),
				verbosity(as<int>(l["verbosity"])) {}
  
  List params() {
    return List::create(Named("epochs") = epochs,
			Named("learning_rate") = learning_rate,
			Named("momentum") = momentum,
			Named("minibatch_size") = minibatch_size,
			Named("l2_reg") = l2_reg,
			Named("shuffle") = shuffle,
			Named("verbosity") = verbosity);
  }
  
  void fit(NumericMatrix X, NumericMatrix y) {
    stochastic_gradient_descent(X, y, w,
				epochs, learning_rate, momentum,
				minibatch_size, l2_reg, shuffle, verbosity);
  }
  
  NumericMatrix coef() {
    return wrap(w);
  }
  
private:
  int epochs, minibatch_size, verbosity;
  double learning_rate, momentum, l2_reg;
  bool shuffle;
  arma::mat w;
  
};

RCPP_MODULE(LinearModels) {
  class_<logistic_regression>("logistic_regression")
    .constructor<List>()
    
    .method("Params", &logistic_regression::params)
    .method("Fit", &logistic_regression::fit)
    .method("Coef", &logistic_regression::coef)
    ;
}
