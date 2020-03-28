library(MASS)
library(ggplot2)
library(reshape2)
require(ggplot2)
require(gridExtra)
require(reshape2)
library(rlist)
require(RnavGraphImageData)

source("miss.saem.qn.R")
source("louis_lr_saem.R")
source("log_reg.R")
source("likelihood_saem.R")
theme_set(theme_bw())
options(digits = 2)

data(digits) #import USPS digits dataabse

#Random matrix plotting (16X16) like USPS database
image(matrix(rexp(256, rate=.1), ncol=16))

N <- 5  # number of images
dataset = digits[,3000:(3000+N)]
for (i in 1:N){
  sample.digit = matrix(dataset[,i], ncol=16,byrow=FALSE)  
  image(t(sample.digit)[,nrow(sample.digit):1])
}

# sample.digit = matrix(digits[,3000], ncol=16,byrow=FALSE)
# image(t(sample.digit)[,nrow(sample.digit):1])


#Hyperparam
p <- ncol(sample.digit)
Gamma.star <- diag(rep(1,p)) # covariance


# mu.star <- rep(0,p)  # mean of the explanatory variables
# Sigma.star <- diag(rep(1,p)) # covariance
# beta.star <- c(1, 1,  0) # coefficients
# beta0.star <- 0 # intercept
# beta.true = c(beta0.star,beta.star)
# X.complete <- matrix(rnorm(N*p), nrow=N)%*%chol(Sigma.star) +
#               matrix(rep(mu.star,N), nrow=N, byrow = TRUE)
# p1 <- 1/(1+exp(-X.complete%*%beta.star-beta0.star))
# y <- as.numeric(runif(N)<p1)



batchsize = 1
nb.epochs <-10
nb.iter <- N/batchsize*nb.epochs

# SAEM
fit.params = miss.saem(X.obs,y,maxruns=nb.iter,k1=0,ll_obs_cal=FALSE, algo = "saem")

#MCEM
# list.mcem = miss.saem(X.obs,y,maxruns=nb.iter,ll_obs_cal=FALSE, algo = "mcem")
# #Incremental MCEM
# list.imcem = miss.saem(X.obs,y,maxruns=nb.iter,ll_obs_cal=FALSE, algo = "imcem", batchsize= batchsize)

#Images of the deformable model using the fitted parameters.


