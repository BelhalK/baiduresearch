require(ggplot2)
require(gridExtra)
require(reshape2)
library(rlist)

source("utils/algos.R")
source("utils/func.R")
source("utils/plots.R")
theme_set(theme_bw())
options(digits = 2)

### IMPORTANT PARAMS ####
n <- 1000
nsim=3
nb.epochs <- 1
########################

K <- n*nb.epochs
weight<-c(0.2, 0.8)
mean <- 0.5
mu<-c(mean,-mean)
sigma<-c(1,1)*1
weight0<-weight
mean0 <- 1.1
mu0<-c(mean0,-mean0)
sigma0<-sigma
seed0=23422

G<-length(mu)
col.names <- c("iteration", paste0("p",1:G), paste0("mu",1:G), paste0("sigma",1:G))
theta<-list(p=weight,mu=mu,sigma=sigma)
theta0<-list(p=weight0,mu=mu0,sigma=sigma0)
x <- matrix(0,nrow=n,ncol=nsim)
mls <- list()
end <- 200

for (j in (1:nsim))
{
  seed <- j*seed0
  set.seed(seed)
  xj<-mixt.simulate(n,weight,mu,sigma)
  x[,j] <- xj

  df <- mixt.em(x[,j], theta0, end)
  p1 = c(rep(df[end,2],(K+1)))
  p2 = c(rep(df[end,3],(K+1)))
  mu1 = c(rep(df[end,4],(K+1)))
  mu2 = c(rep(df[end,5],(K+1)))
  sigma1 = c(rep(df[end,6],(K+1)))
  sigma2 = c(rep(df[end,7],(K+1)))

  ML <- cbind(1:(K+1),p1,p2,mu1,mu2,sigma1,sigma2)
  mls[[j]] <- ML
}

df.em <- vector("list", length=nsim)
df.saem <- vector("list", length=nsim)
df.iemseq <- vector("list", length=nsim)
df.isaem <- vector("list", length=nsim)
df.isaemvr <- vector("list", length=nsim)
df.isaemsaga <- vector("list", length=nsim)


nbr<-1 #nb of replacement
Kem <- nbr*K/n
kiter = 1:K
rho.vr = 1/(n)**(2/3)
rho.saga = 1/(n)**(2/3)

nb.chains <- 2

for (j in (1:nsim))
{	
  cat(j,"/", nsim, "Run\n",sep=" ")
  seed <- j*seed0
  set.seed(seed)
  ML <- mls[[j]]
  
  df <- mixt.em(x[,j], theta0, nb.epochs)
  df[,2:7] <- (df[,2:7] - ML[1:(Kem+1),2:7])^2
  df.em[[j]] <- df
  print('em done')

  df <- mixt.iem.seq(x[,j], theta0, nb.epochs*n/nbr,nbr)
  df[,2:7] <- (df[,2:7] - ML[,2:7])^2
  df.iemseq[[j]] <- df
  print('iemseq done')

  
  df <- mixt.saem(x[,j],theta0, nb.epochs, K1=Kem/2, alpha=0.6, M=1)
  df[,2:7] <- (df[,2:7] - ML[,2:7])^2
  df.saem[[j]] <- df
  print('saem done')

  df <- mixt.isaem(x[,j],theta0, nb.epochs*n/nbr, K1=K/2, alpha=0.6, M=nb.chains,nbr)
  df[,2:7] <- (df[,2:7] - ML[,2:7])^2
  df.isaem[[j]] <- df
  print('isaem done')

  df <- mixt.isaemvr(x[,j],theta0, nb.epochs*n/nbr, K1=K/2, alpha=1, M=nb.chains,nbr, rho.vr)
  df[,2:7] <- (df[,2:7] - ML[,2:7])^2
  df.isaemvr[[j]] <- df
  print('isaemvr done')

  df <- mixt.isaemsaga(x[,j],theta0, nb.epochs*n/nbr, K1=K/2, alpha=0.6, M=nb.chains,nbr, rho.saga)
  df[,2:7] <- (df[,2:7] - ML[,2:7])^2
  df.isaemsaga[[j]] <- df
  print('isaemsaga done')

}

em <- Reduce("+",df.em)/nsim
iemseq <- Reduce("+",df.iemseq)/nsim
saem <- Reduce("+",df.saem)/nsim
isaem <- Reduce("+",df.isaem)/nsim
isaemvr <- Reduce("+",df.isaemvr)/nsim
isaemsaga <- Reduce("+",df.isaemsaga)/nsim

em$algo <- 'EM'
iemseq$algo <- 'IEM'
saem$algo <- 'saem'
isaem$algo <- 'isaem'
isaemvr$algo <- 'isaemvr'
isaemsaga$algo <- 'isaemsaga'

iemseq$iteration <- isaem$iteration*nbr/n
isaem$iteration <- isaem$iteration*nbr/n
isaemvr$iteration <- isaemvr$iteration*nbr/n
isaemsaga$iteration <- isaemsaga$iteration*nbr/n


variance <- rbind(isaem[,c(1,4,8)],
                  isaemsaga[,c(1,4,8)],
                  isaemvr[,c(1,4,8)],
                  iemseq[,c(1,4,8)],
                  em[,c(1,4,8)],
                  saem[,c(1,4,8)])

save(variance, file = "saved/gmm_tts.RData")
write.csv(variance, file = "saved/runtts.csv")

