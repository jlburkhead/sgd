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
y <- matrix(as.numeric(iris$Species == "versicolor"))
multiclass_y <- model.matrix(~Species - 1, data = iris)
X <- model.matrix(Species ~ Sepal.Length + Sepal.Width, data = iris)
```


##### using `stats::glm` for binary classification


```r

coef(glm(y ~ X - 1, family = binomial))
```

```
##  X(Intercept) XSepal.Length  XSepal.Width 
##        8.0928        0.1294       -3.2128
```



##### batch


```r

stochastic_gradient_descent(X, y, 10000, 0.01, 0.95, minibatch_size = nrow(X), 
    shuffle = FALSE)
```

```
##         [,1]
## [1,]  8.0747
## [2,]  0.1305
## [3,] -3.2089
```


##### minibatch


```r

stochastic_gradient_descent(X, y, 10000, 0.01, 0.99, minibatch_size = 10)
```

```
##         [,1]
## [1,]  8.1522
## [2,]  0.1674
## [3,] -3.2101
```


##### stochastic


```r

stochastic_gradient_descent(X, y, 10000, 0.01, 0.99, minibatch_size = 1)
```

```
##         [,1]
## [1,]  8.3860
## [2,]  0.2078
## [3,] -3.2682
```


## multiclass


```r

params <- stochastic_gradient_descent(X, multiclass_y, 10000, 0.01, 0.99, minibatch_size = 10)

params
```

```
##        [,1]    [,2]    [,3]
## [1,]  17.86  8.1521 -14.247
## [2,] -10.16  0.1236   2.634
## [3,]  11.96 -3.2315  -0.737
```

```r

preds <- plogis(X %*% params)  ## sigmoid
preds <- t(apply(preds, 1, function(x) x/sum(x)))

by(preds, iris$Species, colMeans)
```

```
## INDICES: setosa
##      V1      V2      V3 
## 0.86015 0.10953 0.03032 
## -------------------------------------------------------- 
## INDICES: versicolor
##      V1      V2      V3 
## 0.01978 0.56294 0.41727 
## -------------------------------------------------------- 
## INDICES: virginica
##       V1       V2       V3 
## 0.003606 0.366720 0.629674
```

