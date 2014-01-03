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

glm(y ~ X - 1, family = binomial)
```

```
## 
## Call:  glm(formula = y ~ X - 1, family = binomial)
## 
## Coefficients:
##  X(Intercept)  XSepal.Length   XSepal.Width  
##         8.093          0.129         -3.213  
## 
## Degrees of Freedom: 150 Total (i.e. Null);  147 Residual
## Null Deviance:	    208 
## Residual Deviance: 152 	AIC: 158
```

```r

gradient_descent(X, y, 10000, 0.01, 0.99)
```

```
## [1]  8.0764  0.1304 -3.2093
```

```r
stochastic_gradient_descent(X, y, 10000, 0.01, 0.999)
```

```
## [1]  8.1487  0.1143 -3.2322
```


