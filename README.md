sgd
===========





```r

ptm <- proc.time()

library(sgd)
library(rbenchmark)

set.seed(42)

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

batch <- LogisticRegression(epochs = 1500, learning_rate = 0.1, momentum = 0.99, 
    minibatch_size = 0)
batch$Fit(X, y)
batch$Coef()
```

```
##         [,1]
## [1,]  8.1266
## [2,]  0.1289
## [3,] -3.2234
```


##### minibatch


```r

minibatch <- LogisticRegression(epochs = 10000, learning_rate = 0.01, momentum = 0.99, 
    minibatch_size = 10)
minibatch$Fit(X, y)
minibatch$Coef()
```

```
##         [,1]
## [1,]  8.1522
## [2,]  0.1674
## [3,] -3.2101
```


##### stochastic


```r

stochastic <- LogisticRegression(epochs = 10000, learning_rate = 0.01, momentum = 0.99, 
    minibatch_size = 1)
stochastic$Fit(X, y)
stochastic$Coef()
```

```
##         [,1]
## [1,]  8.3860
## [2,]  0.2078
## [3,] -3.2682
```


## multiclass


```r

multiclass <- LogisticRegression(epochs = 10000, learning_rate = 0.01, momentum = 0.99, 
    minibatch_size = 10)
multiclass$Fit(X, multiclass_y)

params <- multiclass$Coef()

preds <- plogis(X %*% params)  ## sigmoid
preds <- t(apply(preds, 1, function(x) x/sum(x)))

by(preds, iris$Species, colMeans)
```

```
## INDICES: setosa
##      V1      V2      V3 
## 0.86041 0.10928 0.03031 
## -------------------------------------------------------- 
## INDICES: versicolor
##      V1      V2      V3 
## 0.01901 0.56334 0.41765 
## -------------------------------------------------------- 
## INDICES: virginica
##       V1       V2       V3 
## 0.003555 0.366699 0.629747
```


# Benchmarks




## Iris benchmarks


```r

data(iris)
set.seed(42)

y <- matrix(as.numeric(iris$Species == "versicolor"))
X <- model.matrix(Species ~ Sepal.Length + Sepal.Width, data = iris)
X[, -1] <- scale(X[, -1])

sgd <- LogisticRegression(epochs = 500, learning_rate = 0.01, momentum = 0.95, 
    minibatch_size = 0)

benchmark(glm = glm(y ~ X - 1, family = binomial), sgd_R = sgd_R(X, y, 500, 
    0.01, 0.95), sgd = sgd$Fit(X, y), replications = 100)
```

```
##    test replications elapsed relative user.self sys.self user.child
## 1   glm          100   0.292    1.000     0.287    0.002          0
## 3   sgd          100   0.435    1.490     0.435    0.000          0
## 2 sgd_R          100   1.598    5.473     1.593    0.003          0
##   sys.child
## 1         0
## 3         0
## 2         0
```


## Test against MNIST data





```r

mnist <- LogisticRegression(momentum = 0.95, minibatch_size = 0)
mnist$Fit(train_X, train_y)

valid_pred <- mnist$Predict_proba(valid_X)
valid_pred <- apply(valid_pred, 1, which.max) - 1
```


missclassification rate: 0.0953


## More benchmarks to come





```r

proc.time() - ptm
```

```
##    user  system elapsed 
##  122.27   20.22  152.77
```

```r
sessionInfo()
```

```
## R version 3.0.2 (2013-09-25)
## Platform: x86_64-apple-darwin10.8.0 (64-bit)
## 
## locale:
## [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
## 
## attached base packages:
## [1] stats     graphics  grDevices utils     datasets  methods   base     
## 
## other attached packages:
## [1] rbenchmark_1.0.0 sgd_0.0.0        Rcpp_0.10.6      knitr_1.4.1     
## 
## loaded via a namespace (and not attached):
## [1] codetools_0.2-8 digest_0.6.3    evaluate_0.4.7  formatR_0.9    
## [5] stringr_0.6.2   tools_3.0.2
```




