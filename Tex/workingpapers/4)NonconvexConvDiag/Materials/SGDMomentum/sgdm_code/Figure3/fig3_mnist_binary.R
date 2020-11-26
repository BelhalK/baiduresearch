# Loads data to run logistic regression on MNIST dataset
# The dataset can be downloaded from
#   http://yann.lecun.com/exdb/mnist/
# * t10k-images.idx3-ubyte
# * t10k-labels.idx1-ubyte
# * train-images.idx3-ubyte
# * train-labels.idx1-ubyte
source("fig3_mnist_load.R")

dat <- load_mnist(dir=fpath)
X_train <- dat$train$x
y_train <- dat$train$y
X_test <- dat$test$x
y_test <- dat$test$y

# Set task to be binary classification on digit 9.
y_train[y_train != 9] <- 0
y_train[y_train == 9] <- 1
y_test[y_test != 9] <- 0
y_test[y_test == 9] <- 1

# Set up generic X, y.
X_train[1,1] <- as.numeric(X_train[1,1])
y_train <- as.numeric(y_train)
#X <- X_train
#y <- y_train

# create format for iSGD
data.mnist <- list(model="binomial", 
                   X_train=X_train, Y_train=y_train,
                   X_test=X_test, Y_test=y_test,
                   glm_link=function(x) expit(x),
                   du_glm_link=function(x) du_expit(x))

data.mnist.scaled <- list(model="binomial", 
                          X_train=scale(X_train), Y_train=y_train,
                          X_test=scale(X_test), Y_test=y_test,
                          glm_link=function(x) expit(x),
                          du_glm_link=function(x) du_expit(x))


# delte extra variables to save memory
rm(dat, X_train, y_train, X_test, y_test)

# Uncomment to save data object
#saveRDS(data.mnist, "data.mnist.rds")
