sgd
===========

[![Build Status](https://travis-ci.org/jlburkhead/sgd.png?branch=master)](https://travis-ci.org/jlburkhead/sgd)





```r

ptm <- proc.time()

library(sgd)
library(rbenchmark)

set.seed(42)

data(iris)
y <- matrix(as.numeric(iris$Species == "versicolor"))
multiclass_y <- model.matrix(~Species - 1, data = iris)
X <- model.matrix(Species ~ Sepal.Length + Sepal.Width - 1, data = iris)
```


##### using `stats::glm` for binary classification


```r

coef(glm(y ~ X, family = binomial))
```

```
##   (Intercept) XSepal.Length  XSepal.Width 
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
## [1,]  8.0974
## [2,]  0.1302
## [3,] -3.2154
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

preds <- multiclass$Predict(X)

by(preds, iris$Species, colMeans)
```

```
## INDICES: setosa
##      V1      V2      V3 
## 0.86038 0.10931 0.03031 
## -------------------------------------------------------- 
## INDICES: versicolor
##     V1     V2     V3 
## 0.0191 0.5633 0.4176 
## -------------------------------------------------------- 
## INDICES: virginica
##       V1       V2       V3 
## 0.003561 0.366701 0.629738
```


# Benchmarks




## Iris benchmarks


```r

data(iris)
set.seed(42)

y <- matrix(as.numeric(iris$Species == "versicolor"))
X <- model.matrix(Species ~ Sepal.Length + Sepal.Width - 1, data = iris)
X <- scale(X)

sgd <- LogisticRegression(epochs = 500, learning_rate = 0.01, momentum = 0.95, 
    minibatch_size = 0)

benchmark(glm = glm(y ~ X - 1, family = binomial), sgd_R = sgd_R(X, y, 500, 
    0.01, 0.95), sgd = sgd$Fit(X, y), replications = 100)
```

```
##    test replications elapsed relative user.self sys.self user.child
## 1   glm          100   0.268    1.055     0.264        0          0
## 3   sgd          100   0.254    1.000     0.252        0          0
## 2 sgd_R          100   2.052    8.079     2.048        0          0
##   sys.child
## 1         0
## 3         0
## 2         0
```


## Test against MNIST data





```r

mnist <- LogisticRegression(momentum = 0.95, minibatch_size = 0)
mnist$Fit(train_X, train_y)

valid_pred <- mnist$Predict_class(valid_X) - 1
```


missclassification rate: 0.0926


## More benchmarks to come





```r

proc.time() - ptm
```

```
##    user  system elapsed 
##  107.06   19.16  105.12
```

```r
sessionInfo()
```

```
## R version 3.0.2 (2013-09-25)
## Platform: x86_64-pc-linux-gnu (64-bit)
## 
## locale:
##  [1] LC_CTYPE=en_US.UTF-8       LC_NUMERIC=C              
##  [3] LC_TIME=en_US.UTF-8        LC_COLLATE=en_US.UTF-8    
##  [5] LC_MONETARY=en_US.UTF-8    LC_MESSAGES=en_US.UTF-8   
##  [7] LC_PAPER=en_US.UTF-8       LC_NAME=C                 
##  [9] LC_ADDRESS=C               LC_TELEPHONE=C            
## [11] LC_MEASUREMENT=en_US.UTF-8 LC_IDENTIFICATION=C       
## 
## attached base packages:
## [1] stats     graphics  grDevices utils     datasets  methods   base     
## 
## other attached packages:
## [1] rbenchmark_1.0.0 sgd_0.0.0        Rcpp_0.10.6      knitr_1.5       
## 
## loaded via a namespace (and not attached):
## [1] codetools_0.2-8 evaluate_0.5.1  formatR_0.10    stringr_0.6.2  
## [5] tools_3.0.2
```




