#include "multilayer_perceptron.hpp"
#include "stochastic_gradient_descent.hpp"


using namespace Rcpp;


base_layer::base_layer(bool use_bias_, 
		       arma::mat (*act_) (arma::mat, arma::mat), 
		       arma::mat (*derivative_) (arma::mat)) {
  use_bias = use_bias_;
  act = act_;
  derivative = derivative_;

}
  
arma::mat base_layer::forward_propagate(arma::mat X) {
  if (w.n_cols == 0) 
    init_weight_(X.n_cols, size);
  a = activate(X);
  return a;
}

arma::mat base_layer::backpropagate(arma::mat delta, 
				    double learning_rate,
				    double momentum) {
  delta_w = momentum * delta_w + (1 - momentum) * learning_rate * incoming_activation.t() * delta;
  w = w - delta_w;
  if (use_bias) {
    delta_bias = momentum * delta_bias + (1 - momentum) * learning_rate * arma::sum(delta, 0);
    bias = bias - delta_bias;
  }
  return delta * w.t() % derivative(incoming_activation);
}

arma::mat base_layer::activate(arma::mat X) {
  incoming_activation = X;
  arma::mat activation_(X.n_cols, size);
  if (use_bias) {
    X = arma::join_horiz(arma::ones(X.n_rows, 1), X);
    arma::mat wb = arma::join_vert(bias, w);
    activation_ = act(X, wb);
  } else {
    activation_ = act(X, w);
  }
  return activation_;
}

arma::mat base_layer::coef() {
  return arma::join_vert(bias, w);
}

void base_layer::init_weight_(int n_in, int n_out) {
  if (w.n_cols == 0) {
    size = n_out;
    w = arma::randn(n_in, n_out);
    delta_w = arma::zeros(n_in, n_out);
    if (use_bias) {
      bias = arma::randn(1, n_out);
      delta_bias = arma::zeros(1, n_out);
    }
  }
}

logistic_layer::logistic_layer() : base_layer(true, sigmoid_activation, d_sigmoid) {};

softmax_layer::softmax_layer() : base_layer(true, softmax_activation, d_sigmoid) {};


mlp::mlp(List l) { 

  n_hidden = as<int>(l["n_hidden"]);
  epochs = as<int>(l["epochs"]);
  minibatch_size = as<int>(l["minibatch_size"]);
  learning_rate = as<double>(l["learning_rate"]);
  momentum = as<double>(l["momentum"]);

}

void mlp::fit(NumericMatrix X, NumericMatrix y) {
  mlp_gradient_descent(X, y, hidden, output, epochs, learning_rate, momentum, minibatch_size, l2_reg, shuffle);
}

NumericMatrix mlp::predict(NumericMatrix X) {
  arma::mat Xm = as< arma::mat >(X);
  return wrap(output.activate(hidden.activate(Xm)));
}

List mlp::coef() {
  return List::create(Named("hidden") = wrap(hidden.coef()),
		      Named("output") = wrap(output.coef()));
}


RCPP_MODULE(MLP) {

  class_<mlp>(".mlp")
    .constructor<List>()
    
    .method("Fit", &mlp::fit)
    .method("Predict", &mlp::predict)
    .method("Coef", &mlp::coef)
    ;

}
