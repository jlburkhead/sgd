library(sgd)

data(iris)

X <- scale(model.matrix(Species ~ . - 1, data = iris))
y <- as.matrix(as.numeric(iris$Species == "versicolor"))

mlp <- MLPClassifier(n_hidden = 100, learning_rate = 0.1, momentum = 0.95, minibatch_size = 0)

system.time(replicate(100, mlp$Fit(X, y)))


by(mlp$Predict(X), iris$Species, colMeans)



untar("mnist.csv.tar.gz")
mnist <- read.csv("mnist.csv")
mnist[-1] <- scale(mnist[-1])

nan <- sapply(mnist, function(x) any(is.nan(x)) )
mnist <- mnist[!nan]

idx <- sample(1:10, nrow(mnist), replace = TRUE)

train <- mnist[idx != 1,]
valid <- mnist[idx == 1,]

train_X <- model.matrix(label ~ . - 1, data = train)
train_y <- model.matrix(~ factor(train$label) - 1)

valid_X <- model.matrix(label ~ . - 1, data = valid)

mlp <- MLPClassifier(epochs = 200, n_hidden = 100, learning_rate = 0.1, momentum = 0.95, minibatch_size = 0)
system.time(mlp$Fit(train_X, train_y))

prob <- mlp$Predict(valid_X)
pred <- apply(prob, 1, which.max) - 1

mean(pred != valid$label)
