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
			   shuffle(as<bool>(l["shuffle"])) {}
  
  List params() const {
    return List::create(Named("epochs") = epochs,
			Named("learning_rate") = learning_rate,
			Named("momentum") = momentum,
			Named("minibatch_size") = minibatch_size,
			Named("l2_reg") = l2_reg,
			Named("shuffle") = shuffle);
  }
  
  NumericMatrix coef() const { return wrap(w); }
  
  NumericMatrix link(const NumericMatrix X) {
    arma::mat Xm = as< arma::mat >(X);
    return wrap(activation(Xm, w));
  }
  
  virtual NumericMatrix predict(const NumericMatrix X) { return 0; }
  virtual void fit(const NumericMatrix X, const NumericMatrix y) {}
  
protected:
  
  void fit_(const NumericMatrix X, const NumericMatrix y, arma::mat (*act) (arma::mat, arma::mat)) {
    stochastic_gradient_descent(X, y, w, act, epochs, learning_rate, momentum, minibatch_size, l2_reg, shuffle);
  }
  
  NumericMatrix predict_(const NumericMatrix X, arma::mat (*act) (arma::mat, arma::mat)) {
    arma::mat Xm = as< arma::mat >(X);
    return wrap(act(Xm, w));
  }
  
  int epochs, minibatch_size;
  double learning_rate, momentum, l2_reg;
  bool shuffle;
  arma::mat w;

};

class linear_regression : public base_regressor  {
public:
  linear_regression(List l_) : base_regressor(l_) {}
  
  void fit(const NumericMatrix X, const NumericMatrix y) { 
    fit_(X, y, activation);
  }
  
  NumericMatrix predict(const NumericMatrix X) {
    return predict_(X, activation);
  }
  
};


class logistic_regression : public base_regressor {
public:
  logistic_regression(List l_) : base_regressor(l_) {}

  void fit(const NumericMatrix X, const NumericMatrix y) { 
    fit_(X, y, sigmoid_activation);
  }
  
  // TODO: a method to predict class
  NumericMatrix predict(const NumericMatrix X) {
    return predict_(X, sigmoid_activation);
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
    .method("Link", &base_regressor::link)
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
