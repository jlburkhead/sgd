#include <RcppArmadillo.h>

#include "sigmoid.hpp"
#include "stochastic_gradient_descent.hpp"
#include "activation.hpp"

using namespace Rcpp;


class base_regressor {
public:
  base_regressor(List l) : epochs(as<int>(l["epochs"])),
			   learning_rate(as<double>(l["learning_rate"])),
			   momentum(as<double>(l["momentum"])),
			   minibatch_size(as<int>(l["minibatch_size"])),
			   l2_reg(as<double>(l["l2_reg"])),
			   shuffle(as<bool>(l["shuffle"])),
			   verbosity(as<int>(l["verbosity"])) {}
  
  List params() const {
    return List::create(Named("epochs") = epochs,
			Named("learning_rate") = learning_rate,
			Named("momentum") = momentum,
			Named("minibatch_size") = minibatch_size,
			Named("l2_reg") = l2_reg,
			Named("shuffle") = shuffle,
			Named("verbosity") = verbosity);
  }
  
  NumericMatrix coef() const { return wrap(w); }
  
  virtual NumericMatrix predict(const NumericMatrix X) { return 0; }
  virtual void fit(const NumericMatrix X, const NumericMatrix y) {}
  
protected:
  int epochs, minibatch_size, verbosity;
  double learning_rate, momentum, l2_reg;
  bool shuffle;
  arma::mat w;

};

class linear_regression : public base_regressor  {
public:
  linear_regression(List l_) : base_regressor(l_) {}

  void fit(const NumericMatrix X, const NumericMatrix y) { 
      stochastic_gradient_descent(X, y, w, activation, epochs, learning_rate, momentum, 
				  minibatch_size, l2_reg, shuffle, verbosity);
  }
  
  NumericMatrix predict(const NumericMatrix X) {
    arma::mat Xm = as< arma::mat >(X);
    return wrap(activation(Xm, w));
  }
  
};


class logistic_regression : public base_regressor {
public:
  logistic_regression(List l_) : base_regressor(l_) {}

  void fit(const NumericMatrix X, const NumericMatrix y) { 
      stochastic_gradient_descent(X, y, w, sigmoid_activation, epochs, learning_rate, momentum, 
				  minibatch_size, l2_reg, shuffle, verbosity);
  }
  
  // TODO: a method to predict class
  NumericMatrix predict(const NumericMatrix X) {
    arma::mat Xm = as< arma::mat >(X);
    return wrap(activation(Xm, w));
  }

  NumericMatrix predict_proba(const NumericMatrix X) {
    arma::mat Xm = as< arma::mat >(X);
    return wrap(sigmoid(Xm * w));
  }
  
};



RCPP_MODULE(LinearModels) {
  class_<base_regressor>("base_regressor")
    .constructor<List>()
    
    .method("Params", &base_regressor::params)
    .method("Coef", &base_regressor::coef)
    .method("Fit", &base_regressor::fit)
    .method("Predict", &base_regressor::predict)
    ;
    
  class_<logistic_regression>("logistic_regression")
    .derives<base_regressor>("base_regressor")
    .constructor<List>()
    
    .method("Predict_proba", &logistic_regression::predict_proba)
    ;

  class_<linear_regression>("linear_regression")
    .derives<base_regressor>("base_regressor")
    .constructor<List>()
    
    ;

}
