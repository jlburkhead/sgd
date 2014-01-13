#include <RcppArmadillo.h>

#include "activation.hpp"
#include "sigmoid.hpp"

using namespace Rcpp;


class base_layer {
public:
  base_layer(bool use_bias_, 
	     arma::mat (*act_) (arma::mat, arma::mat), 
	     arma::mat(*derivative_) (arma::mat)) :  use_bias(use_bias_),
						     act(act_),
						     derivative(derivative_) {}
  
  arma::mat forward_propagate(arma::mat X) {
    if (w.n_cols == 0) {
      w = arma::randn(X.n_cols, size);
      if (use_bias)
	bias = arma::randn(1, size);
    }
    if (use_bias) {
      X = arma::join_horiz(arma::ones(X.n_rows, 1), X);
      arma::mat wb = arma::join_vert(bias, w);
      a = act(X, wb);
    } else {
      a = act(X, w);
    }
    return a;
  }

  arma::mat backpropagate(arma::mat delta, double learning_rate) {
    w = w - learning_rate * a.t() * delta;
    if (use_bias)
      bias = bias - learning_rate * arma::mean(delta, 0);
    return delta * w.t() % derivative(a);
  }

protected:
  int size;
  bool use_bias;
  arma::mat w, a, bias;
  arma::mat (*act) (arma::mat, arma::mat);
  arma::mat (*derivative) (arma::mat);

};

class logistic_layer : public base_layer {
public:
  logistic_layer() : base_layer(true, sigmoid_activation, d_sigmoid) {}
};

class softmax_layer : public base_layer {
public:
  softmax_layer() : base_layer(true, softmax_activation, d_sigmoid) {}
};


class mlp2 {
public:
  mlp2(int n_in, int n_hidden, int n_out, bool use_bias, double learning_rate_) : learning_rate(learning_rate_) {}
  
private:
  double learning_rate;
  logistic_layer hidden;
  softmax_layer output;
};

class mlp {
public:
  mlp(NumericMatrix X, NumericMatrix y, int n_hidden, double learning_rate_) : learning_rate(learning_rate_) {
    Xm = as< arma::mat >(X);
    ym = as< arma::mat >(y);
    w1 = arma::randn(Xm.n_cols, n_hidden);
    w2 = arma::randn(n_hidden, ym.n_cols);
    b1 = arma::randn(1, n_hidden);
    b2 = arma::randn(1, ym.n_cols);
  }
  
  
  void fit(int epochs) {
    for (int e = 0; e < epochs; e++) {
      
      arma::mat hidden_activation = sigmoid_activation(arma::join_horiz(arma::ones(Xm.n_rows, 1), Xm), arma::join_vert(b1, w1));
      arma::mat output_activation = softmax_activation(arma::join_horiz(arma::ones(Xm.n_rows, 1), hidden_activation), arma::join_vert(b2, w2));
      
      arma::mat delta_output = ym - output_activation;
      arma::mat delta_hidden = delta_output * w2.t() % sigmoid(hidden_activation) % (1 - sigmoid(hidden_activation));

      w2 = w2 - hidden_activation.t() * delta_output * learning_rate;
      w1 = w1 - Xm.t() * delta_hidden * learning_rate;
      
      b1 = b1 - arma::mean(delta_hidden, 0) * learning_rate;
      b2 = b2 - arma::mean(delta_output, 0) * learning_rate;

    }
  }

  NumericMatrix predict(NumericMatrix X) {
    arma::mat Xm = as< arma::mat >(X);

    return wrap(sigmoid(Xm * w1 * w2));
  }

private:
  double learning_rate;
  arma::mat Xm, ym, w1, w2, b1, b2;

};
  

RCPP_MODULE(MLP) {
  class_<mlp>("mlp")
    .constructor<NumericMatrix, NumericMatrix, int, double>()
    
    .method("Fit", &mlp::fit)
    .method("Predict", &mlp::predict)
    ;
}
