# load("usps1.RData")
library(MASS)
library(ggplot2)
library(reshape2)
require(ggplot2)
require(gridExtra)
require(reshape2)
library(rlist)
require(RnavGraphImageData)
source("src/saem.R")
source("src/vrsaem.R")
source("src/fisaem.R")
source("src/utils.R")
theme_set(theme_bw())
options(digits = 2)

#import USPS digits dataabse
data(digits) 

nb <- 20  # number of images
images = digits[,5000:(5000+nb)]

# # plots some digits images
# for (i in 1:nb){
#   sample.digit = matrix(images[,i], ncol=16,byrow=FALSE)  
#   image(t(sample.digit)[,nrow(sample.digit):1])
# }

template.model<-function(z, xi,p,landmarks.p,landmarks.g) { 
  zi<-z
  ypred = matrix(NA,nrow=p,ncol=p)
  phi <- as.list(numeric(p^2))
  dim(phi) <- c(p,p)
  sigma.g = 1
  sigma.p = 1
  for (m in 1:p){
  	for (j in 1:p){
      #Image Coordinate Standard
  		x.ind = 2*m/p-1
  		y.ind = 2*j/p-1
  		rep.coord = matrix(c(x.ind,y.ind), nrow=1)
	   	coord <- t(apply(rep.coord, 2, rep, kg))
      coord
      	#deformation computation
  		kernel.deformation = exp(-(coord - landmarks.g)^2/(2*sigma.g))
	   	phi[[m,j]]= colSums(kernel.deformation)%*%t(zi)
      	#template computation
      	coord.template = rep.coord - phi[[m,j]]
      	rep.coord.template <- t(apply(coord.template, 2, rep, kp))
      	kernel.template = exp(-(rep.coord.template - landmarks.p)^2/(2*sigma.p))
  		template = colSums(kernel.template)%*%xi
  		ypred[m,j] = template
  	}
  } 
  return(ypred)
}

#Hyperparam
p <- ncol(sample.digit) #dimension of the input
kp <- 5 #dimension of the parameter of the template
kg <- 6 #dimension of the random effects
Gamma.star <- diag(rep(1,kg)) # covariance

batchsize = 1
nb.epochs <- 5
N <- ncol(images)
nb.iter <- N/batchsize*nb.epochs
nb.mcmc <- 4

#first stage of SAEM
K1 = 0
rho.vr = 1/N**(2/3)
rho.saga = 1/N**(2/3)

#fixed landmarks points
landmarks.p = matrix(rnorm(2*kp,mean = 0, sd = 0.5),ncol=kp) #of template
landmarks.g = matrix(rnorm(2*kg,mean = 0, sd = 0.5),ncol=kg) #of deformation

# SAEM
fit.saem = tts.saem(images,kp,kg,landmarks.p,landmarks.g, template.model,
  maxruns=nb.epochs,nmcmc = nb.mcmc,k1=K1,algo = "saem", batchsize=batchsize)
fit.inc.saem = tts.saem(images,kp,kg,landmarks.p,landmarks.g, template.model,
  maxruns=nb.iter,nmcmc = nb.mcmc,k1=K1,algo = "isaem", batchsize=batchsize)
fit.vr.saem = vrsaem(images,kp,kg,landmarks.p,landmarks.g, template.model,
  maxruns=nb.iter,nmcmc = nb.mcmc,k1=K1,algo = "vrsaem", batchsize=batchsize,rho.vr)
fit.fi.saem = fisaem(images,kp,kg,landmarks.p,landmarks.g, template.model,
  maxruns=nb.iter,nmcmc = nb.mcmc,k1=K1,algo = "fisaem", batchsize=batchsize,rho.saga)


#PLOTS
# dim=1
# saem = fit.saem$seqxi[dim,]
# isaem = fit.inc.saem$seqxi[dim,]

# #PER ITERATION
# x = 1:length(saem)
# df <- data.frame(x,saem, isaem)
# df2 <- melt(df ,  id.vars = 'x', variable.name = 'algo')
# ggplot(df2, aes(x,value)) + geom_line(aes(colour = algo))


# #PER EPOCHS
# epochs = seq(1,nb.iter,N/batchsize)
# x = 2:(nb.epochs+1)
# saem.ep <- saem[x]
# isaem.ep <- isaem[(epochs+1)]
# df <- data.frame(x,saem.ep,isaem.ep)
# df2 <- melt(df ,  id.vars = 'x', variable.name = 'algo')
# ggplot(df2, aes(x,value)) + geom_line(aes(colour = algo))



## generate new images with fitted params
#mean template
p=sqrt(nrow(images)) #dimension of the input
temp = 0
for (i in 1:nb){
  sample.digit = matrix(images[,i], ncol=16,byrow=FALSE)  
  temp = temp +sample.digit
}
meantemp = temp/nb
image(t(meantemp)[,nrow(meantemp):1], axes = FALSE, col = grey(seq(0, 1, length = 256)))

#Generated samples per epoch
#Batch 
newsamples <- list()
for (i in 1:nb.epochs){
  xi <- fit.saem$seqxi[,i]
  Gamma <- fit.saem$seqgamma[[i]]
  sigma <- fit.saem$seqsigma[,i]
  chol.omega <- chol(Gamma)
  z.new <- matrix(rnorm(2*kg),ncol=kg)%*%chol.omega
  newsamples[[i]]<-template.model(z.new, xi, p,landmarks.p,landmarks.g) #generated digit  
}

for (i in 1:nb.epochs){
  final = newsamples[[i]] + meantemp
  image(t(final)[,nrow(final):1], axes = FALSE, col = grey(seq(0, 1, length = 256)))
}


epochs = seq(1,nb.iter,N/batchsize)
#Incremental
newsamples.inc <- list()
for (i in epochs){
  xi <- fit.inc.saem$seqxi[,i]
  Gamma <- fit.inc.saem$seqgamma[[i]]
  sigma <- fit.inc.saem$seqsigma[,i]
  chol.omega <- chol(Gamma)
  z.new <- matrix(rnorm(2*kg),ncol=kg)%*%chol.omega
  newsamples.inc[[i]]<-template.model(z.new, xi, p,landmarks.p,landmarks.g) #generated digit  
}

for (i in epochs){
  final = newsamples.inc[[i]] + meantemp
  image(t(final)[,nrow(final):1], axes = FALSE, col = grey(seq(0, 1, length = 256)))
}

#last one for both
image(t(tail(newsamples,1)[[1]] + meantemp)[,nrow(tail(newsamples,1)[[1]] + meantemp):1], axes = FALSE, col = grey(seq(0, 1, length = 256)))
image(t(tail(newsamples.inc,1)[[1]] + meantemp)[,nrow(tail(newsamples.inc,1)[[1]] + meantemp):1], axes = FALSE, col = grey(seq(0, 1, length = 256)))


#VR 
newsamples.vr <- list()
for (i in 1:nb.epochs){
  xi <- fit.vr.saem$seqxi[,i]
  Gamma <- fit.vr.saem$seqgamma[[i]]
  sigma <- fit.vr.saem$seqsigma[,i]
  chol.omega <- chol(Gamma)
  z.new <- matrix(rnorm(2*kg),ncol=kg)%*%chol.omega
  newsamples.vr[[i]]<-template.model(z.new, xi, p,landmarks.p,landmarks.g) #generated digit  
}

for (i in 1:nb.epochs){
  final = newsamples.vr[[i]] + meantemp
  image(t(final)[,nrow(final):1], axes = FALSE, col = grey(seq(0, 1, length = 256)))
}


#FI
newsamples.fi <- list()
for (i in epochs){
  xi <- fit.fi.saem$seqxi[,i]
  Gamma <- fit.fi.saem$seqgamma[[i]]
  sigma <- fit.fi.saem$seqsigma[,i]
  chol.omega <- chol(Gamma)
  z.new <- matrix(rnorm(2*kg),ncol=kg)%*%chol.omega
  newsamples.fi[[i]]<-template.model(z.new, xi, p,landmarks.p,landmarks.g) #generated digit  
}

for (i in epochs){
  final = newsamples.fi[[i]] + meantemp
  image(t(final)[,nrow(final):1], axes = FALSE, col = grey(seq(0, 1, length = 256)))
}

image(t(tail(newsamples,1)[[1]] + meantemp)[,nrow(tail(newsamples,1)[[1]] + meantemp):1], axes = FALSE, col = grey(seq(0, 1, length = 256)))
image(t(tail(newsamples.inc,1)[[1]] + meantemp)[,nrow(tail(newsamples.inc,1)[[1]] + meantemp):1], axes = FALSE, col = grey(seq(0, 1, length = 256)))
image(t(tail(newsamples.vr,1)[[1]] + meantemp)[,nrow(tail(newsamples.vr,1)[[1]] + meantemp):1], axes = FALSE, col = grey(seq(0, 1, length = 256)))
image(t(tail(newsamples.fi,1)[[1]] + meantemp)[,nrow(tail(newsamples.fi,1)[[1]] + meantemp):1], axes = FALSE, col = grey(seq(0, 1, length = 256)))

# save.image("usps2.RData")