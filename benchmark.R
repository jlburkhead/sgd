library(sgd)
library(rbenchmark)

sigmoid <- function(x) 1 / (1 + exp(-x))
activation <- function(X, w) sigmoid(X %*% w)
cross_entropy <- function(y, h) -(y * log(h) + (1 - y) * log(1 - h))
gradient <- function(X, y, h) t(X) %*% (h - y)

sgd_R <- function(X, y, epochs, learning_rate, momentum)
  {
    
    n <- nrow(X)
    p <- ncol(X)
    k <- ncol(y)

    w <- matrix(rnorm(p * k), p, k)
    delta_w <- matrix(0, p, k)

    for (e in 1:epochs) {

      h <- activation(X, w)
      g <- gradient(X, y, h)

      delta_w <- momentum * delta_w + (1 - momentum) * learning_rate * g
      w <- w - delta_w

      ce <- cross_entropy(y, h)
      
    }

    return(w)
    
  }

data(iris)

y <- matrix(as.numeric(iris$Species == "versicolor"))
X <- model.matrix(Species ~ Sepal.Length + Sepal.Width, data = iris)
X[,-1] <- scale(X[,-1])


benchmark(
  glm = glm(y ~ X - 1, family = binomial),
  R = sgd_R(X, y, 500, 0.01, 0.95),
  sgd = stochastic_gradient_descent(X, y, 500, 0.01, 0.95, nrow(X), shuffle = FALSE),
  replications = 100
  )

##   test replications elapsed relative user.self sys.self user.child sys.child
## 1  glm          100   0.262    1.000     0.261    0.001          0         0
## 2    R          100   3.488   13.313     3.484    0.003          0         0
## 3  sgd          100   0.437    1.668     0.432    0.000          0         0


## mnist
library(ggplot2)
theme_set(theme_bw())
set.seed(1)

d <- read.csv("train.csv")
d[-1] <- scale(d[-1]) ## quick

train_idx <- sample(1:10, nrow(d), replace = TRUE)

train <- d[train_idx != 1,]

nan <- sapply(train, function(x) any(is.nan(x)) )

train <- train[!nan]
valid <- d[train_idx == 1, !nan]

train_X <- model.matrix(label ~ ., data = train)
train_y <- model.matrix(~ factor(train$label) - 1)

params <- stochastic_gradient_descent(train_X, train_y, 100, 0.01, 0.99, minibatch_size = nrow(train_X), shuffle = FALSE)

valid_X <- model.matrix(label ~ ., data = valid)
valid_pred <- plogis(valid_X %*% params)
valid_pred <- apply(valid_pred, 1, which.max) - 1

cat("missclassification rate: ", mean(valid_pred != valid$label), "\n")

benchmark(
  glm = glm(train_y[1:5000,1] ~ train_X[1:5000,], family = binomial),
  sgd = stochastic_gradient_descent(train_X[1:5000,], train_y[1:5000, 1, drop = FALSE], 100, 0.01, 0.99, nrow(train_X), shuffle = FALSE),
  replications = 1
  )



## binary
Rprof(filename = "sgd_binary.out", memory.profiling = TRUE)
params <- stochastic_gradient_descent(train_X[1:5000,], train_y[1:5000,1, drop = FALSE], 100, 0.01, 0.99)
Rprof(NULL)

sgd_binary_prof <- summaryRprof("sgd_binary.out", memory = "tseries")[-1,]

Rprof(filename = "glm_binary.out", memory.profiling = TRUE)
Sys.sleep(5)
mod <- glm(train_y[1:5000, 1] ~ train_X[1:5000,], family = binomial)
Rprof(NULL)

glm_binary_prof <- summaryRprof("glm_binary.out", memory = "tseries")[-1,]

binary_prof <- rbind(sgd_binary_prof, glm_binary_prof)
binary_prof$fun <- c(rep("sgd", nrow(sgd_binary_prof)), rep("glm", nrow(glm_binary_prof)))

p <- ggplot(binary_prof, aes(x = 1:nrow(binary_prof), y = vsize.large, colour = fun)) + labs(x = "t", y = "memory usage") + geom_path()

## multiclass
Rprof(filename = "sgd_multiclass.out", memory.profiling = TRUE)
params <- stochastic_gradient_descent(train_X, train_y, 100, 0.01, 0.99)
Rprof(NULL)

sgd_prof <- summaryRprof("sgd_multiclass.out", memory = "tseries")

