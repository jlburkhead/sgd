#include <RcppArmadillo.h>

#include "activation.hpp"
#include "sigmoid.hpp"

using namespace Rcpp;


// class base_layer {
// public:
//   base_layer(int n_in, int n_out, arma::mat (*act_) (arma::mat, arma::mat)) : act(act_) {
//     w = arma::randn(n_in, n_out);
//     a(n_out, 1);
//   }
  
//   arma::mat forward_propagate(arma::mat X) {
//     a = act(X, w);
//     return a;
//   }

//   arma::mat backpropagate(arma::mat delta, double learning_rate) {
//     w = w - learning_rate * a % delta;
//   }

// protected:
//   arma::mat w, a;
//   arma::mat (*act) (arma::mat, arma::mat);

// };

// class logistic_layer : public base_layer {
// public:
//   logistic_layer(int n_in, int n_out) : base_layer(n_in, n_out, sigmoid_activation) {}


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
