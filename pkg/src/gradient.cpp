#include "gradient.hpp"

arma::mat gradient(arma::mat X, arma::mat y, arma::mat h) {
  return X.t() * (h - y);
}
