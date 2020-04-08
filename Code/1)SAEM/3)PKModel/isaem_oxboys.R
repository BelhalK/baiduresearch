library("mlxR")
library("rlist")
library("psych")
library("coda")
library("Matrix")
library(abind)
require(ggplot2)
require(gridExtra)
require(reshape2)

source('R/aaa_generics.R') 
source('R/compute_LL.R') 
source('R/func_aux.R') 
source('R/func_distcond.R') 
source('R/func_FIM.R')
source('R/func_plots.R') 
source('R/func_simulations.R') 
source('R/main.R')
source('R/main_initialiseMainAlgo.R') 
source('R/main_mstep.R') 
source('R/SaemixData.R')
source('R/SaemixModel.R')
source('R/SaemixRes.R')
source('R/SaemixObject.R') 
source('R/zzz.R')

#new
source('R/main_estep.R')
source('R/main_estep_fi.R')
source('R/main_mstep_vr.R') 
source('R/main_mstep_fi.R') 
source('R/mixtureFunctions.R')
source('R/plots.R')


oxboys.saemix<-read.table( "data/oxboys.saemix.tab",header=T,na=".")
saemix.data<-saemixData(name.data=oxboys.saemix,header=TRUE,
  name.group=c("Subject"),name.predictors=c("age"),name.response=c("height"),
  units=list(x="yr",y="cm"))


growth.linear<-function(psi,id,xidep) {
# input:
#   psi : matrix of parameters (2 columns, base and slope)
#   id : vector of indices 
#   xidep : dependent variables (same nb of rows as length of id)
# returns:
#   a vector of predictions of length equal to length of id
  x<-xidep[,1]
  base<-psi[id,1]
  slope<-psi[id,2]
  f<-base+slope*x
  return(f)
}
saemix.model<-saemixModel(model=growth.linear,description="Linear model",type="structural",
  psi0=matrix(c(140,1),ncol=2,byrow=TRUE,dimnames=list(NULL,c("base","slope"))),
  transform.par=c(1,0),covariance.model=matrix(c(1,1,1,1),ncol=2,byrow=TRUE), 
  error.model="constant")


K1 = 200
K2 = 30
iterations = 0:(K1+K2-1)
end = K1+K2
batchsize25 = 25
batchsize50 = 50 

seed0=3456

nchains = 1
gamma = 1

### BATCH ###
options<-list(seed=39546,map=F,fim=F,ll.is=F,save.graphs=FALSE,nb.chains = nchains,nbiter.mcmc = c(2,2,2,0), 
  nbiter.saemix = c(K1,K2),nbiter.sa=0,displayProgress=FALSE,nbiter.burn =0, 
  map.range=c(0), nb.replacement=100,sampling='randomiter',gamma=gamma, algo="full")
fit.ref<-saemix(saemix.model,saemix.data,options)
fit.ref <- data.frame(fit.ref$param)
fit.ref <- cbind(iterations, fit.ref[-1,])

### INCREMENTAL ###
options.50<-list(seed=seed0,map=F,fim=F,ll.is=F,save.graphs=FALSE,nb.chains = nchains, nbiter.mcmc = c(2,2,2,0), 
                          nbiter.saemix = c(K1,K2),displayProgress=FALSE, map.range=c(0),nbiter.sa=0,
                          nbiter.burn =0, nb.replacement=50,sampling='seq',gamma=gamma,algo="minibatch")
fit.50<-saemix(saemix.model,saemix.data,options.50)
fit.50 <- data.frame(fit.50$param)
fit.50 <- cbind(iterations, fit.50[-1,])

### Variance Reduced ###
options.vr.50<-list(seed=seed0,map=F,fim=F,ll.is=F,save.graphs=FALSE,nb.chains = nchains, nbiter.mcmc = c(2,2,2,0), 
                          nbiter.saemix = c(K1,K2),displayProgress=FALSE, map.range=c(0),nbiter.sa=0,
                          nbiter.burn =0, nb.replacement=50,sampling='randomiter',gamma=gamma,algo="vr", rho =0.01)
fit.vr.50<-saemix(saemix.model,saemix.data,options.vr.50)
fit.vr.50 <- data.frame(fit.vr.50$param)
fit.vr.50 <- cbind(iterations, fit.vr.50[-1,])

# ### Fast Iterative ###
options.fi.50<-list(seed=seed0,map=F,fim=F,ll.is=F,save.graphs=FALSE,nb.chains = nchains, nbiter.mcmc = c(2,2,2,0), 
                          nbiter.saemix = c(K1,K2),displayProgress=FALSE, map.range=c(0),nbiter.sa=0,
                          nbiter.burn =0, nb.replacement=50,sampling='seq',gamma=gamma,algo="fi", rho =0.1)
fit.fi.50<-saemix(saemix.model,saemix.data,options.fi.50)
# fit.fi.50 <- data.frame(fit.fi.50$param)
# fit.fi.50 <- cbind(iterations, fit.fi.50[-1,])

fit.ref.scaled <- fit.ref
# fit.25.scaled <- fit.25
# fit.75.scaled <- fit.75
# fit.25.scaled$iterations = fit.25.scaled$iterations*0.25
# fit.75.scaled$iterations = fit.75.scaled$iterations*0.75
fit.50.scaled <- fit.50
fit.50.scaled$iterations = fit.50.scaled$iterations*0.5
fit.50.vr.scaled <- fit.vr.50
fit.50.vr.scaled$iterations = fit.50.vr.scaled$iterations*0.5
graphConvMC_5(fit.ref.scaled,fit.50.scaled,fit.50.scaled,fit.50.scaled,fit.50.vr.scaled)
#black, blue, red, yellow, pink





# options.75<-list(seed=seed0,map=F,fim=F,ll.is=F,save.graphs=FALSE,nb.chains = nchains, 
#   nbiter.mcmc = c(2,2,2,0), nbiter.saemix = c(K1,K2),displayProgress=FALSE, map.range=c(0),
#   nbiter.sa=0,nbiter.burn =0, nb.replacement=75,sampling='randomiter',gamma=gamma,algo="minibatch")
# fit.75<-saemix(saemix.model,saemix.data,options.75)
# fit.75 <- data.frame(fit.75$param)
# fit.75 <- cbind(iterations, fit.75[-1,])

# options.25<-list(seed=seed0,map=F,fim=F,ll.is=F,save.graphs=FALSE,nb.chains = nchains, 
#   nbiter.mcmc = c(2,2,2,0), nbiter.saemix = c(K1,K2),displayProgress=FALSE, map.range=c(0),
#   nbiter.sa=0,nbiter.burn =0, nb.replacement=25,sampling='seq',gamma=gamma,algo="minibatch")
# fit.25<-saemix(saemix.model,saemix.data,options.25)
# fit.25 <- data.frame(fit.25$param)
# fit.25 <- cbind(iterations, fit.25[-1,])