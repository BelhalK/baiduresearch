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

# #Random matrix plotting (16X16) like USPS database
# image(matrix(rexp(256, rate=.1), ncol=16))

nb <- 5  # number of images
images = digits[,3000:(3000+nb)]

# #plots some digits images
# for (i in 1:nb){
#   sample.digit = matrix(images[,i], ncol=16,byrow=FALSE)  
#   image(t(sample.digit)[,nrow(sample.digit):1])
# }


#Hyperparam
p <- ncol(sample.digit) #dimension of the input
kp <- 5 #dimension of the parameter of the template
kg <- 6 #dimension of the random effects
Gamma.star <- diag(rep(1,kg)) # covariance



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


batchsize = 1
nb.epochs <-3
N <- ncol(images)
nb.iter <- N/batchsize*nb.epochs
nb.mcmc <- 4

#first stage of SAEM
K1 = 0

rho.vr = 1/N**(2/3)
rho.saga = 1/N**(2/3)


# SAEM
# fit.saem = tts.saem(images,kp,kg, template.model,maxruns=nb.iter,nmcmc = nb.mcmc,k1=K1,algo = "saem", batchsize=batchsize)
# fit.inc.saem = tts.saem(images,kp,kg, template.model,maxruns=nb.iter,nmcmc = nb.mcmc,k1=K1,algo = "isaem", batchsize=batchsize)
# fit.vr.saem = vrsaem(images,kp,kg, template.model,maxruns=nb.iter,nmcmc = nb.mcmc,k1=K1,algo = "vrsaem", batchsize=batchsize,rho.vr)
fit.fi.saem = fisaem(images,kp,kg, template.model,maxruns=nb.iter,nmcmc = nb.mcmc,k1=K1,algo = "fisaem", batchsize=batchsize,rho.saga)



# fit.saem$seqxi
# fit.saem$seqgamma[[nb.iter]]
# fit.saem$seqsigma

# #PLOTS
# dim = 1
# saem = fit.saem$seqxi[dim,1:nb.iter]
# isaem = fit.inc.saem$seqxi[dim,1:nb.iter]
# x = 1:length(saem)
# df <- data.frame(x,saem)
# ggplot(data=df)+
#   geom_line(mapping=aes(y=saem,x= x,color="saem"),size=0.5 ) +
#   geom_line(mapping=aes(y=isaem,x= x,color="isaem"),size=0.5 ) +
#   scale_color_manual(values = c(
#     'saem' = 'darkblue', 'isaem' = 'red')) +
#   labs(color = 'Algo')+ ylab("xi")


# #PER EPOCHS
# epochs = seq(1,nb.iter,N/batchsize)
# x = 2:(nb.epochs+1)
# saem.ep <- saem[x]
# isaem.ep <- isaem[(epochs+1)]
# df <- data.frame(x,saem.ep,isaem.ep)

# ggplot(data=df)+
#   geom_line(mapping=aes(y=saem.ep,x= x,color="saem"),size=0.5 ) +
#   geom_line(mapping=aes(y=isaem.ep,x= x,color="isaem"),size=0.5) +
#   scale_color_manual(values = c(
#     'saem' = 'darkblue','isaem' = 'red')) +
#   labs(color = 'Algo')


## generate new images with fitted params
landmarks.p = matrix(rnorm(2*kp),ncol=kp) #of template
landmarks.g = matrix(rnorm(2*kg),ncol=kg) #of deformation

xi <- fit.saem$seqxi[,nb.iter]
Gamma <- fit.saem$seqgamma[[nb.iter]]
sigma <- fit.saem$seqsigma[,nb.iter]

chol.omega <- chol(Gamma)
z.new <- matrix(rnorm(2*kg),ncol=kg)%*%chol.omega
p=sqrt(nrow(images)) #dimension of the input

new.sample<-template.model(z.new, xi, p,landmarks.p,landmarks.g) #generated digit
image(t(new.sample)[,nrow(new.sample):1]) #display generated digit


