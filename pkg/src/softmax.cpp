#include "softmax.hpp"

using namespace arma;

void softmax(mat& X) {
  for (int i = 0; i < X.n_rows; i++)
    X.row(i) = X.row(i) / accu(X.row(i));
}
  
