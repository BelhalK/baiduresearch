load("Rdata/testFI.RData")
library("mlxR")
library("rlist")
library("psych")
library("coda")
library("Matrix")
library(abind)
require(ggplot2)
require(gridExtra)
require(reshape2)

source('R/SaemixData.R')
source('R/SaemixModel.R')
source('R/SaemixRes.R')
source('R/SaemixObject.R') 
source('R/zzz.R')



# ### Fast Iterative ###
options.fi.50<-list(seed=seed0,map=F,fim=F,ll.is=F,save.graphs=FALSE,nb.chains = nchains,
  nbiter.mcmc = iter.mcmc, nbiter.saemix = c(K1,K2),displayProgress=FALSE,map.range=c(0),
  nbiter.sa=0,nbiter.burn =0, nb.replacement=50,sampling='randomiter',algo="fi", rho =0.1)
fit.fi.50<-saemix.fi(saemix.model,saemix.data,options.fi.50)

fit.fi.50 <- data.frame(fit.fi.50$param)
fit.fi.50 <- cbind(iterations, fit.fi.50[-1,])
fit.50.fi.scaled <- fit.fi.50
fit.50.fi.scaled$iterations = fit.50.fi.scaled$iterations*0.5
graphConvMC_5(fit.ref.scaled,fit.50.scaled,fit.50.scaled,fit.50.fi.scaled,fit.50.vr.scaled)
