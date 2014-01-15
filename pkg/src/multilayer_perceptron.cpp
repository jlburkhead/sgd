#include "multilayer_perceptron.hpp"

using namespace Rcpp;


base_mlp::base_mlp(List l) : n_hidden(as<int>(l["n_hidden"])),
			     epochs(as<int>(l["epochs"])),
			     learning_rate(as<double>(l["learning_rate"])),
			     momentum(as<double>(l["momentum"])),
			     minibatch_size(as<int>(l["minibatch_size"])),
			     l2_reg(as<double>(l["l2_reg"])),
			     shuffle(as<bool>(l["shuffle"])),
			     t(0) {}

List base_mlp::coef() {
  return List::create(Named("hidden") = wrap(arma::join_vert(b1, w1)),
		      Named("output") = wrap(arma::join_vert(b2, w2)));
}

void base_mlp::init_fit_() {
  
  w1 = arma::randn(n_features, n_hidden);
  delta_w1 = arma::randn(n_features, n_hidden);
  w2 = arma::randn(n_hidden, n_outputs);
  delta_w2 = arma::randn(n_hidden, n_outputs);
  b1 = arma::randn(1, n_hidden);
  delta_b1 = arma::randn(1, n_hidden);
  b2 = arma::randn(1, n_outputs);
  delta_b2 = arma::randn(1, n_outputs);
  
}

void base_mlp::init_param_() {
  
  hidden_func = sigmoid;
  hidden_derivative = d_sigmoid;
  if (n_outputs > 1)
    output_func = softmax;
  else
    output_func = sigmoid;
}

void base_mlp::fit_(arma::mat X, arma::mat y) {
  
  int n = X.n_rows;
  n_outputs = y.n_cols;
  n_features = X.n_cols;

  if (t == 0 || w1.n_cols == 0) {
    init_fit_();
    init_param_();
  }
  
  if (minibatch_size == 0)
    minibatch_size = n;
  int minibatches = (n - 1) / minibatch_size + 1;
  
  if (minibatch_size != n && shuffle)
    shuffle_matrix(X, y);
  
  for (int i = 0; i < minibatches; i++) {
    int start = i * minibatch_size;
    int end = start + minibatch_size - 1;
    if (end > n - 1)
      end = n - 1;
    arma::span s(start, end);

    arma::mat X_minibatch = X(s, arma::span::all);
    arma::mat y_minibatch = y(s, arma::span::all);

    int this_minibatch_size = X_minibatch.n_rows; // don't use this
    
    hidden_activation = hidden_func(X_minibatch * w1 + arma::repmat(b1, this_minibatch_size, 1));
    arma::mat output_activation = output_func(hidden_activation * w2 + arma::repmat(b2, this_minibatch_size, 1));

    arma::mat delta_output = output_activation - y_minibatch;
    arma::mat delta_hidden = delta_output * w2.t() % hidden_derivative(hidden_activation);

    arma::mat w1_gradient = X_minibatch.t() * delta_hidden + l2_reg * w1 / this_minibatch_size ;
    arma::mat w2_gradient = hidden_activation.t() * delta_output + l2_reg * w2 / this_minibatch_size;
    
    arma::mat b1_gradient = arma::mean(delta_hidden, 0);
    arma::mat b2_gradient = arma::mean(delta_output, 0);

    delta_w1 = momentum * delta_w1 + (1 - momentum) * learning_rate * w1_gradient;
    delta_w2 = momentum * delta_w2 + (1 - momentum) * learning_rate * w2_gradient;
    delta_b1 = momentum * delta_b1 + (1 - momentum) * learning_rate * b1_gradient;
    delta_b2 = momentum * delta_b2 + (1 - momentum) * learning_rate * b2_gradient;
    
    w1 = w1 - delta_w1;
    w2 = w2 - delta_w2;
    b1 = b1 - delta_b1;
    b2 = b2 - delta_b2;
    
  }
  
  t++;

}
  
  
arma::mat base_mlp::predict_(arma::mat X) {

  int n = X.n_rows;
  
  arma::mat ha = hidden_func(X * w1 + arma::repmat(b1, n, 1));
  arma::mat oa = output_func(ha * w2 + arma::repmat(b2, n, 1));
  
  return oa;

}

mlp_classifier::mlp_classifier(List l) : base_mlp(l) {}

void mlp_classifier::fit(NumericMatrix X, NumericMatrix y) {
  arma::mat Xm = as< arma::mat >(X);
  arma::mat ym = as< arma::mat >(y);
  
  for (int e = 0; e < epochs; e++)
    fit_(Xm, ym);
  
}

NumericMatrix mlp_classifier::predict(NumericMatrix X) {
  arma::mat Xm = as< arma::mat >(X);
  return wrap(predict_(Xm));
}


RCPP_MODULE(MLP) {
  class_<base_mlp>(".base_mlp")
    .constructor<List>()
    
    .method("Coef", &base_mlp::coef)
    ;

  class_<mlp_classifier>(".mlp_classifier")
    .derives<base_mlp>(".base_mlp")
    .constructor<List>()
    
    .method("Fit", &mlp_classifier::fit)
    .method("Predict", &mlp_classifier::predict)
    ;
}
