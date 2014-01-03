#include "cross_entropy.hpp"

arma::mat cross_entropy(arma::mat y, arma::mat h) {
  return -(y % log(h) + (1 - y) % log(1 - h));
}
