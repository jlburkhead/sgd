#include <RcppArmadillo.h>

#include "sigmoid.hpp"
#include "stochastic_gradient_descent.hpp"
#include "activation.hpp"
#include "arma_utils.hpp"

using namespace Rcpp;


class base_regressor {
public:
  base_regressor(List l) : epochs(as<int>(l["epochs"])),
			   learning_rate(as<double>(l["learning_rate"])),
			   momentum(as<double>(l["momentum"])),
			   minibatch_size(as<int>(l["minibatch_size"])),
			   l2_reg(as<double>(l["l2_reg"])),
			   shuffle(as<bool>(l["shuffle"])),
			   fit_intercept(as<bool>(l["fit_intercept"])) {}
  
  List params() const {
    return List::create(Named("epochs") = epochs,
			Named("learning_rate") = learning_rate,
			Named("momentum") = momentum,
			Named("minibatch_size") = minibatch_size,
			Named("l2_reg") = l2_reg,
			Named("shuffle") = shuffle,
			Named("fit_intercept") = fit_intercept);
  }
  
  NumericMatrix coef() const { return wrap(w); }
  
  NumericMatrix link(const NumericMatrix X) {
    arma::mat Xm = as< arma::mat >(X);
    if (fit_intercept)
      Xm = arma::join_horiz(arma::ones(Xm.n_rows, 1), Xm);
    return wrap(activation(Xm, w));
  }
  
  virtual NumericMatrix predict(const NumericMatrix X) { 
    stop("Not implemented");
    return 0; 
  }
  virtual void fit(const NumericMatrix X, const NumericMatrix y) {
    stop("Not implemented");
  }
  
protected:
  
  void fit_(const NumericMatrix X, const NumericMatrix y, arma::mat (*act) (arma::mat, arma::mat)) {
    arma::mat Xm = as< arma::mat >(X);
    arma::mat ym = as< arma::mat >(y);
    if (fit_intercept)
      Xm = arma::join_horiz(arma::ones(Xm.n_rows, 1), Xm);
    stochastic_gradient_descent(Xm, ym, w, act, epochs, learning_rate, momentum, minibatch_size, l2_reg, shuffle, fit_intercept);
  }
  
  NumericMatrix predict_(const NumericMatrix X, arma::mat (*act) (arma::mat, arma::mat)) {
    arma::mat Xm = as< arma::mat >(X);
    if (fit_intercept)
      Xm = arma::join_horiz(arma::ones(Xm.n_rows, 1), Xm);
    return wrap(act(Xm, w));
  }
  
  int epochs, minibatch_size;
  double learning_rate, momentum, l2_reg;
  bool shuffle, fit_intercept;
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
  
  NumericMatrix predict(const NumericMatrix X) {
    NumericMatrix pred = predict_(X, sigmoid_activation);
    if (pred.ncol() > 1) {
      arma::mat predm = as< arma::mat >(pred);
      row_normalize(predm);
      pred = wrap(predm);
    }
    return pred;
  }

  IntegerVector predict_class(const NumericMatrix X) {
    int n = X.nrow();
    arma::mat prob = as< arma::mat>(predict(X));
    int k = prob.n_cols;
    IntegerVector out(n);
    
    if (k == 1) {
      for (int i = 0; i < n; i++)
	out[i] = round(prob(i));
      return out;
    } else {
      out = row_max(prob) + 1;
      return out;
    }

  }
  
};


class poisson_regression : public base_regressor {
public:
  poisson_regression(List l_) : base_regressor(l_) {}

  void fit(const NumericMatrix X, const NumericMatrix y) {
    fit_(X, y, poisson_activation);
  }

  NumericMatrix predict(const NumericMatrix X) {
    return predict_(X, poisson_activation);
  }

};

RCPP_MODULE(LinearModels) {
  class_<base_regressor>(".base_regressor")
    .constructor<List>()
    
    .method("Params", &base_regressor::params)
    .method("Coef", &base_regressor::coef)
    .method("Link", &base_regressor::link)
    .method("Fit", &base_regressor::fit)
    .method("Predict", &base_regressor::predict)
    ;
    
  class_<logistic_regression>(".logistic_regression")
    .derives<base_regressor>(".base_regressor")
    .constructor<List>()
    
    .method("Predict_class", &logistic_regression::predict_class)
    ;

  class_<linear_regression>(".linear_regression")
    .derives<base_regressor>(".base_regressor")
    .constructor<List>()
    
    ;

  class_<poisson_regression>(".poisson_regression")
    .derives<base_regressor>(".base_regressor")
    .constructor<List>()

    ;

}
