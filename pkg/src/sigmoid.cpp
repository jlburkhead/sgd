#include "sigmoid.hpp"

arma::mat sigmoid(arma::mat x) {
  return 1 / (1 + exp(-x));
}
