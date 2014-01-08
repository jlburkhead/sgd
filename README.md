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

stochastic_gradient_descent(X, y, 1500, 0.1, 0.99, minibatch_size = nrow(X), 
    shuffle = FALSE)
```

```
##         [,1]
## [1,]  8.0908
## [2,]  0.1296
## [3,] -3.2125
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
## [1,]  18.27  8.1521 -14.247
## [2,] -10.27  0.1236   2.634
## [3,]  12.00 -3.2315  -0.737
```

```r

preds <- plogis(X %*% params)  ## sigmoid
preds <- t(apply(preds, 1, function(x) x/sum(x)))

by(preds, iris$Species, colMeans)
```

```
## INDICES: setosa
##      V1      V2      V3 
## 0.86029 0.10939 0.03031 
## -------------------------------------------------------- 
## INDICES: versicolor
##      V1      V2      V3 
## 0.01937 0.56315 0.41748 
## -------------------------------------------------------- 
## INDICES: virginica
##       V1       V2       V3 
## 0.003578 0.366708 0.629714
```


# Benchmarks




## Iris benchmarks


```r

data(iris)
set.seed(42)

y <- matrix(as.numeric(iris$Species == "versicolor"))
X <- model.matrix(Species ~ Sepal.Length + Sepal.Width, data = iris)
X[, -1] <- scale(X[, -1])

benchmark(glm = glm(y ~ X - 1, family = binomial), sgd_R = sgd_R(X, y, 500, 
    0.01, 0.95), sgd = stochastic_gradient_descent(X, y, 500, 0.01, 0.95, nrow(X), 
    shuffle = FALSE), replications = 100)
```

```
##    test replications elapsed relative user.self sys.self user.child
## 1   glm          100   0.283    1.000     0.280        0          0
## 3   sgd          100   0.766    2.707     0.768        0          0
## 2 sgd_R          100   1.909    6.746     1.904        0          0
##   sys.child
## 1         0
## 3         0
## 2         0
```


## Test against MNIST data





```r

params <- stochastic_gradient_descent(train_X, train_y, 100, 0.01, 0.95, minibatch_size = nrow(train_X))

valid_pred <- plogis(valid_X %*% params)
valid_pred <- apply(valid_pred, 1, which.max) - 1
```


missclassification rate: 0.0948


## More benchmarks


```r

benchmark(glm = glm(train_y[1:5000, 1] ~ train_X[1:5000, ], family = binomial), 
    sgd_batch = stochastic_gradient_descent(train_X[1:5000, ], train_y[1:5000, 
        1, drop = FALSE], 100, 0.01, 0.99, nrow(train_X), shuffle = FALSE), 
    sgd_mb100 = stochastic_gradient_descent(train_X[1:5000, ], train_y[1:5000, 
        1, drop = FALSE], 100, 0.01, 0.99, 100), sgd_stochastic = stochastic_gradient_descent(train_X[1:5000, 
        ], train_y[1:5000, 1, drop = FALSE], 100, 0.01, 0.99, 1), replications = 1)
```

```
##             test replications elapsed relative user.self sys.self
## 1            glm            1  88.908   45.177    88.882    0.112
## 2      sgd_batch            1   1.968    1.000     1.964    0.000
## 3      sgd_mb100            1  10.569    5.370    10.549    0.000
## 4 sgd_stochastic            1  15.043    7.644    15.013    0.000
##   user.child sys.child
## 1          0         0
## 2          0         0
## 3          0         0
## 4          0         0
```



```r

proc.time() - ptm
```

```
##    user  system elapsed 
##   339.9    17.9   337.6
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
## [1] rbenchmark_1.0.0 sgd_0.0.0        knitr_1.5       
## 
## loaded via a namespace (and not attached):
## [1] evaluate_0.5.1 formatR_0.10   stringr_0.6.2  tools_3.0.2
```




