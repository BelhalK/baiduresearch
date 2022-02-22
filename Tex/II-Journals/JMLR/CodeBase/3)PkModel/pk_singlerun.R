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
source('R/main_mstep_fi2.R') 
source('R/main_fi.R') 
source('R/mixtureFunctions.R')




model1cpt<-function(psi,id,xidep) { 
  dose<-xidep[,1]
  time<-xidep[,2]  
  ka<-psi[id,1]
  V<-psi[id,2]
  Cl<-psi[id,3]
  k <- Cl/V
  ypred<-dose*ka/(V*(ka-k))*(exp(-k*time)-exp(-ka*time))
  return(ypred)
}

model <- inlineModel("


[INDIVIDUAL]
input = {ka_pop, V_pop, Cl_pop, omega_ka, omega_V, omega_Cl}
DEFINITION:
ka = {distribution=lognormal, reference=ka_pop, sd=omega_ka}
V  = {distribution=lognormal, reference=V_pop,  sd=omega_V }
Cl = {distribution=lognormal, reference=Cl_pop, sd=omega_Cl}


[LONGITUDINAL]
input = {ka, V, Cl,a}
EQUATION:
C = pkmodel(ka,V,Cl)
DEFINITION:
y = {distribution=normal, prediction=C, sd=a}
")

N=500

param   <- c(
  ka_pop  = 2,    omega_ka  = 0.3,
  V_pop   = 10,   omega_V   = 0.2,
  Cl_pop  = 1,    omega_Cl  = 0.3, a =1)
  
res <- simulx(model     = model,
              parameter = param,
              treatment = list(time=0, amount=100),
              group     = list(size=N, level='individual'),
              output    = list(name='y', time=seq(1,5,by=1)))


 warfarin.saemix <- res$y
 warfarin.saemix$amount <- 100

writeDatamlx(res, result.file = "data/inc_data.csv")

 saemix.data<-saemixData(name.data=warfarin.saemix,header=TRUE,sep=" ",na=NA, name.group=c("id"),
  name.predictors=c("amount","time"),name.response=c("y"), name.X="time")

 model1cpt<-function(psi,id,xidep) { 
    dose<-xidep[,1]
    time<-xidep[,2]  
    Tlag<-psi[id,1]
    ka<-psi[id,2]
    V<-psi[id,3]
    Cl<-psi[id,4]
    k<-Cl/V
    dt <- pmax(time-Tlag, 0)
    ypred<-dose*ka/(V*(ka-k))*(exp(-k*dt)-exp(-ka*dt))
    return(ypred)
  }

  saemix.model<-saemixModel(model=model1cpt,description="warfarin",type="structural"
    ,psi0=matrix(c(0.2,3,10,2),ncol=4,byrow=TRUE, dimnames=list(NULL, c("Tlag","ka","V","Cl"))),
    transform.par=c(1,1,1,1),omega.init=matrix(c(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),ncol=4,byrow=TRUE),
    covariance.model=matrix(c(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),ncol=4, 
    byrow=TRUE),covariate.model=matrix(c(0,0,1,1),ncol=4,byrow=TRUE),error.model="constant")

K1 = 200
K2 = 2
iter.mcmc = c(2,2,2,0)

iterations = 0:(K1+K2-1)
end = K1+K2
nchains = 1


### BATCH ###
options<-list(seed=39546,map=F,fim=F,ll.is=F,save.graphs=FALSE,nb.chains = nchains,
  nbiter.mcmc = iter.mcmc, nbiter.saemix = c(K1,K2),nbiter.sa=0,displayProgress=FALSE,
  nbiter.burn =0, map.range=c(0), nb.replacement=100,sampling='randomiter', algo="full")
fit.ref<-saemix(saemix.model,saemix.data,options)
fit.ref <- data.frame(fit.ref$param)
fit.ref <- cbind(iterations, fit.ref[-1,])

### INCREMENTAL ###
options.50<-list(seed=seed0,map=F,fim=F,ll.is=F,save.graphs=FALSE,nb.chains = nchains, 
  nbiter.mcmc = iter.mcmc, nbiter.saemix = c(K1,K2),displayProgress=FALSE, map.range=c(0),
  nbiter.sa=0,nbiter.burn =0, nb.replacement=50,sampling='seq',algo="minibatch")
fit.50<-saemix(saemix.model,saemix.data,options.50)
fit.50 <- data.frame(fit.50$param)
fit.50 <- cbind(iterations, fit.50[-1,])

### Variance Reduced ###
options.vr.50<-list(seed=seed0,map=F,fim=F,ll.is=F,save.graphs=FALSE,nb.chains = nchains,
  nbiter.mcmc = iter.mcmc, nbiter.saemix = c(K1,K2),displayProgress=FALSE, map.range=c(0),
  nbiter.sa=0,nbiter.burn =0, nb.replacement=50,sampling='randomiter',algo="vr", rho =0.1)
fit.vr.50<-saemix(saemix.model,saemix.data,options.vr.50)
fit.vr.50 <- data.frame(fit.vr.50$param)
fit.vr.50 <- cbind(iterations, fit.vr.50[-1,])

fit.ref.scaled <- fit.ref
fit.50.scaled <- fit.50
fit.50.scaled$iterations = fit.50.scaled$iterations*0.5
fit.50.vr.scaled <- fit.vr.50
fit.50.vr.scaled$iterations = fit.50.vr.scaled$iterations*0.5


# ### Fast Iterative ###
options.fi.50<-list(seed=seed0,map=F,fim=F,ll.is=F,save.graphs=FALSE,nb.chains = nchains,
  nbiter.mcmc = iter.mcmc, nbiter.saemix = c(K1,K2),displayProgress=FALSE,map.range=c(0),
  nbiter.sa=0,nbiter.burn =0, nb.replacement=50,sampling='randomiter',algo="fi", rho =0.1)
fit.fi.50<-saemix.fi(saemix.model,saemix.data,options.fi.50)

fit.fi.50 <- data.frame(fit.fi.50$param)
fit.fi.50 <- cbind(iterations, fit.fi.50[-1,])
fit.50.fi.scaled <- fit.fi.50
fit.50.fi.scaled$iterations = fit.50.fi.scaled$iterations*0.5
