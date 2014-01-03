sgd
===========



```r
set.seed(42)
library(sgd)
```

```
## Loading required package: Rcpp
## Loading required package: RcppArmadillo
```

```r

data(iris)
y <- as.numeric(iris$Species == "versicolor")
X <- model.matrix(Species ~ Sepal.Length + Sepal.Width, data = iris)

coef(glm(y ~ X - 1, family = binomial))
```

```
##  X(Intercept) XSepal.Length  XSepal.Width 
##        8.0928        0.1294       -3.2128
```

```r

gradient_descent(X, y, 10000, 0.01, 0.99)
```

```
## [1]  8.0764  0.1304 -3.2093
```

```r
stochastic_gradient_descent(X, y, 10000, 0.01, 0.99, minibatch_size = 10)
```

```
## [1]  8.1487  0.1119 -3.2339
```


