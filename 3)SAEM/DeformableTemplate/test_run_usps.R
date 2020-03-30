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
images = digits[,3000:(3000+N)]

for (i in 1:N){
  sample.digit = matrix(images[,i], ncol=16,byrow=FALSE)  
  image(t(sample.digit)[,nrow(sample.digit):1])
}


# sample.digit = matrix(digits[,3000], ncol=16,byrow=FALSE)
# image(t(sample.digit)[,nrow(sample.digit):1])


#Hyperparam
p <- ncol(sample.digit) #dimension of the input
kp <- 5 #dimension of the parameter of the template
kg <- 5 #dimension of the random effects
Gamma.star <- diag(rep(1,kg)) # covariance



# #Random EFFECTS
# chol.omega.z<-try(chol(Gamma.star))
# z1 <- matrix(rnorm(2*kg),ncol=kg)%*%chol.omega.z
# z <- list(z1, z1) #random effects (2 X kg)


# landmarks.p = matrix(rnorm(2*kp),ncol=kp)
# landmarks.g = matrix(rnorm(2*kg),ncol=kg)


##TEST OUTSIDE THE FUNCTION 
# m = 2
# j=3
# x.ind = 2*m/p-1
# y.ind = 2*j/p-1
# rep.coord = matrix(c(x.ind,y.ind), nrow=1)
# coord <- t(apply(rep.coord, 2, rep, kg))
# diff = coord - landmarks.g

# kernel.deformation = exp(-(coord - landmarks.g)^2/2)
# colSums(kernel.deformation)%*%t(z[[1]])
# kernel.deformation
##TEST OUTSIDE THE FUNCTION 



template.model<-function(z, xi,id,p) { 
  zi<-z[[id]]
  
  kernel.g = matrix(NA,nrow=1,ncol=p)
  ypred = matrix(NA,nrow=p,ncol=p)

  phi <- as.list(numeric(p^2))
  dim(phi) <- c(p,p)
  
  sigma.g = 1
  sigma.p = 1

  for (m in 1:p){
  	for (j in 1:p){
  		x.ind = 2*m/p-1
  		y.ind = 2*j/p-1
  		rep.coord = matrix(c(x.ind,y.ind), nrow=1)
	   	coord <- t(apply(rep.coord, 2, rep, kg))
  		
  		kernel.deformation = exp(-(coord - landmarks.g)^2/(2*sigma.g))
	   	phi[[i,j]]= colSums(kernel.deformation)%*%t(zi)
  		
  		coord.template = coord - phi[[i,j]]
  		kernel.template = exp(-(coord.template - landmarks.p)^2/(2*sigma.p))

  		template = colSums(kernel.template)%*%t(xi)

  		ypred[i,j] = template
  	}

  } 

  return(ypred)
}


batchsize = 1
nb.epochs <-10
nb.iter <- N/batchsize*nb.epochs

# SAEM
fit.params = miss.saem(images,kp,kg, template.model,maxruns=nb.iter,k1=0,ll_obs_cal=FALSE, algo = "saem")


#MCEM
# list.mcem = miss.saem(X.obs,y,maxruns=nb.iter,ll_obs_cal=FALSE, algo = "mcem")
# #Incremental MCEM
# list.imcem = miss.saem(X.obs,y,maxruns=nb.iter,ll_obs_cal=FALSE, algo = "imcem", batchsize= batchsize)

#Images of the deformable model using the fitted parameters.


