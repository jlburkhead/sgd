#include <RcppArmadillo.h>

using namespace Rcpp;

class LogisticRegression {
public:
  LogisticRegression(int epochs_, 
		     double learning_rate_, 
		     double momentum_,
		     int minibatch_size_,
		     double l2_reg_,
		     bool shuffle_,
		     int verbosity_) : epochs(epochs_),
				       learning_rate(learning_rate_),
				       momentum(momentum_),
				       minibatch_size(minibatch_size_),
				       l2_reg(l2_reg_),
				       shuffle(shuffle_),
				       verbosity(verbosity_) {}
  
  
  int epochs, minibatch_size, verbosity;
  double learning_rate, momentum, l2_reg;
  bool shuffle;
  arma::mat w, delta_w;

};

RCPP_MODULE(LinearModels) {
  class_<LogisticRegression>("LogisticRegression")
    .constructor<int, double, double, int, double, bool, int>()
    
    .field("epochs", &LogisticRegression::epochs)
    ;
}
