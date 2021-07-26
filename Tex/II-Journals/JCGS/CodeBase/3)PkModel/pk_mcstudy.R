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

source('R/main.R')
source('R/main_fi.R')

seed0=3456


K1 = 100
K2 = 20
iter.mcmc = c(2,2,2,0)

iterations = 0:(K1+K2-1)
end = K1+K2
nchains = 1



options<-list(seed=39546,map=F,fim=F,ll.is=F,save.graphs=FALSE,nb.chains = nchains,
    nbiter.mcmc = iter.mcmc, nbiter.saemix = c(K1,K2),nbiter.sa=0,displayProgress=FALSE,
    nbiter.burn =0, map.range=c(0), nb.replacement=100,sampling='randomiter', algo="full")
options.50<-list(seed=seed0,map=F,fim=F,ll.is=F,save.graphs=FALSE,nb.chains = nchains, 
    nbiter.mcmc = iter.mcmc, nbiter.saemix = c(K1,K2),displayProgress=FALSE, map.range=c(0),
    nbiter.sa=0,nbiter.burn =0, nb.replacement=50,sampling='seq',algo="minibatch")
options.vr.50<-list(seed=seed0,map=F,fim=F,ll.is=F,save.graphs=FALSE,nb.chains = nchains,
    nbiter.mcmc = iter.mcmc, nbiter.saemix = c(K1,K2),displayProgress=FALSE, map.range=c(0),
    nbiter.sa=0,nbiter.burn =0, nb.replacement=50,sampling='randomiter',algo="vr", rho =0.1)
options.fi.50<-list(seed=seed0,map=F,fim=F,ll.is=F,save.graphs=FALSE,nb.chains = nchains,
    nbiter.mcmc = iter.mcmc, nbiter.saemix = c(K1,K2),displayProgress=FALSE,map.range=c(0),
    nbiter.sa=0,nbiter.burn =0, nb.replacement=50,sampling='randomiter',algo="fi", rho =0.1)


error.batch <- 0
error.inc <- 0
error.vr <- 0
error.fi <- 0


replicate = 50
for (m in 1:replicate){
  print(m)
  # pk.saemix <- readDatamlx(datafile = paste("data/pk_mcstudy_2.csv", sep=""),  header   = c('id','time','y','amount'))$y
  pk.saemix <-readDatamlx(datafile = paste("data/data_pk/pk_mcstudy_", m, ".csv", sep=""),  header   = c('id','time','y','amount'))$y
  trt <- read.table("design2/treatment.txt", header = TRUE) 
originalId<- read.table('design2/originalId.txt', header=TRUE) 
individualCovariate<- read.table('design2/individualCovariate.txt', header = TRUE) 
  individualCovariate$wt <- log(individualCovariate$wt/70)
  treat <- trt[,c(1,3)]
  covandtreat <- merge(individualCovariate ,treat,by="id")
  pk.saemix <- merge(covandtreat ,pk.saemix[pk.saemix$time!=0,1:3],by="id")

  saemix.data<-saemixData(name.data=pk.saemix,header=TRUE,sep=" ",na=NA, name.group=c("id"),
    name.predictors=c("amount","time"),name.response=c("y"), name.X="time", name.covariates=c("wt"),units=list(x="kg",
    covariates=c("kg/ha")))


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



  ### BATCH ###
  fit.ref<-saemix(saemix.model,saemix.data,options)
  fit.ref <- data.frame(fit.ref$param)
  error.batch <- error.batch + (fit.ref[,2]-1)^2
  fit.ref <- cbind(iterations, fit.ref[-1,])
  ### INCREMENTAL ###
  fit.50<-saemix(saemix.model,saemix.data,options.50)
  fit.50 <- data.frame(fit.50$param)
  error.inc <- error.inc + (fit.50[,2]-1)^2
  fit.50 <- cbind(iterations, fit.50[-1,])
  ### Variance Reduced ###
  fit.vr.50<-saemix(saemix.model,saemix.data,options.vr.50)
  fit.vr.50 <- data.frame(fit.vr.50$param)
  error.vr <- error.vr + (fit.vr.50[,2]-1)^2
  fit.vr.50 <- cbind(iterations, fit.vr.50[-1,])
  # ### Fast Iterative ###
  fit.fi.50<-saemix.fi(saemix.model,saemix.data,options.fi.50)
  fit.fi.50 <- data.frame(fit.fi.50$param)
  error.fi <- error.fi + (fit.fi.50[,2]-1)^2
  fit.fi.50 <- cbind(iterations, fit.fi.50[-1,])
}


error.batch.rep <- cbind(iterations, error.batch[-1]/replicate)
error.inc.rep <- cbind(iterations, error.inc[-1]/replicate)
error.vr.rep <- cbind(iterations, error.vr[-1]/replicate)
error.fi.rep <- cbind(iterations, error.fi[-1]/replicate)

error.batch.df <- data.frame(error.batch.rep)
error.inc.df <- data.frame(error.inc.rep)
error.vr.df <- data.frame(error.vr.rep)
error.fi.df <- data.frame(error.fi.rep)

error.batch.scaled <- error.batch.df
error.inc.scaled <- error.inc.df
error.inc.scaled$iterations = error.inc.scaled$iterations*0.5
error.vr.scaled <- error.vr.df
error.vr.scaled$iterations = error.vr.scaled$iterations*0.5
error.fi.scaled <- error.fi.df
error.fi.scaled$iterations = error.fi.scaled$iterations*0.5


error.batch.df <- data.frame(error.batch.rep)
error.inc.df <- data.frame(error.inc.rep)
error.vr.df <- data.frame(error.vr.rep)
error.fi.df <- data.frame(error.fi.rep)

test.batch <- error.batch.df[rep(seq_len(nrow(error.batch.df)), each = 2), ]
