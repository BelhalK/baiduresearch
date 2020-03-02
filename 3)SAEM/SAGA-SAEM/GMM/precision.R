require(ggplot2)
require(gridExtra)
require(reshape2)
library(rlist)

source("utils/algos.R")
source("utils/func.R")
source("utils/plots.R")
theme_set(theme_bw())
options(digits = 2)

n <- 1000
nb.epochs <- 10
K <- n*nb.epochs
nsim=5

weight<-c(0.2, 0.8)
mean <- 0.5
mu<-c(mean,-mean)
sigma<-c(1,1)*1


weight0<-weight
mean0 <- 1.1
mu0<-c(mean0,-mean0)
sigma0<-sigma
seed0=23422


# ylim <- c(0.15, 0.5, 0.4)
ylim <- c(0.3)

M <- 1

G<-length(mu)
col.names <- c("iteration", paste0("p",1:G), paste0("mu",1:G), paste0("sigma",1:G))
theta<-list(p=weight,mu=mu,sigma=sigma)
theta0<-list(p=weight0,mu=mu0,sigma=sigma0)
# theta0<-theta

x <- matrix(0,nrow=n,ncol=nsim)
mls <- list()
end <- 200
for (j in (1:nsim))
{

  print(j)
  seed <- j*seed0
  set.seed(seed)
  xj<-mixt.simulate(n,weight,mu,sigma)
  x[,j] <- xj
  

  df <- mixt.em(x[,j], theta0, end)
  a1 = c(rep(df[end,2],(K+1)))
  a2 = c(rep(df[end,3],(K+1)))
  b1 = c(rep(df[end,4],(K+1)))
  b2 = c(rep(df[end,5],(K+1)))
  d1 = c(rep(df[end,6],(K+1)))
  d2 = c(rep(df[end,7],(K+1)))

  ML <- cbind(1:(K+1),a1,a2,b1,b2,d1,d2)
  mls[[j]] <- ML
}


nbr<-1

print('EM')
dem <- NULL
df.em <- vector("list", length=nsim)
dsaem <- NULL
df.saem <- vector("list", length=nsim)
# diemseq <- NULL
# df.iemseq <- vector("list", length=nsim)
disaem <- NULL
df.isaem <- vector("list", length=nsim)
disaemvr <- NULL
df.isaemvr <- vector("list", length=nsim)
disaemsaga <- NULL
df.isaemsaga <- vector("list", length=nsim)

Kem <- nbr*K/n
kiter = 1:K
rho.vr = 0.01
rho.saga = 0.01

nb.chains <- 30

for (j in (1:nsim))
{

  print(j)
  seed <- j*seed0
  set.seed(seed)
  ML <- mls[[j]]
  print("ML calculation done")

  df <- mixt.em(x[,j], theta0, nb.epochs)
  df[,2:7] <- (df[,2:7] - ML[1:(Kem+1),2:7])^2
  # df[,2:7] <- (df[,2:7] - ML[,2:7])^2
  df$rep <- j
  dem <- rbind(dem,df)
  df$rep <- NULL
  df.em[[j]] <- df
  print('em done')

  df <- mixt.saem(x[,j],theta0, nb.epochs, K1=Kem/2, alpha=0.6, M=1)
  df[,2:7] <- (df[,2:7] - ML[,2:7])^2
  df$rep <- j
  dsaem <- rbind(dsaem,df)
  df$rep <- NULL
  df.saem[[j]] <- df
  print('saem done')

  df <- mixt.isaem(x[,j],theta0, nb.epochs*n/nbr, K1=K/2, alpha=0.6, M=nb.chains,nbr)
  df[,2:7] <- (df[,2:7] - ML[,2:7])^2
  df$rep <- j
  disaem <- rbind(disaem,df)
  df$rep <- NULL
  df.isaem[[j]] <- df
  print('isaem done')

  df <- mixt.isaemvr(x[,j],theta0, nb.epochs*n/nbr, K1=K/2, alpha=0.6, M=nb.chains,nbr, rho.vr)
  df[,2:7] <- (df[,2:7] - ML[,2:7])^2
  df$rep <- j
  disaemvr <- rbind(disaemvr,df)
  df$rep <- NULL
  df.isaemvr[[j]] <- df
  print('isaemvr done')


  df <- mixt.isaemsaga(x[,j],theta0, nb.epochs*n/nbr, K1=K/2, alpha=0.6, M=nb.chains,nbr, rho.saga)
  df[,2:7] <- (df[,2:7] - ML[,2:7])^2
  df$rep <- j
  disaemsaga <- rbind(disaemsaga,df)
  df$rep <- NULL
  df.isaemsaga[[j]] <- df
  print('isaemsaga done')


}


# dem[,2:7] <- dem[,2:7]^2
em <- NULL
em <- dem[dem$rep==1,]

if (nsim>2) {
   for (j in (2:nsim))
	{
	  em[,2:7] <- em[,2:7]+dem[dem$rep==j,2:7]
	}
}
em[,2:7] <- 1/nsim*em[,2:7]
em[,9]<-NULL

saem <- NULL
saem <- dsaem[dsaem$rep==1,]

if (nsim>2) {
   for (j in (2:nsim))
  {
    saem[,2:7] <- saem[,2:7]+dsaem[dsaem$rep==j,2:7]
  }
}
saem[,2:7] <- 1/nsim*saem[,2:7]
saem[,9]<-NULL



isaem <- NULL
isaem <- disaem[disaem$rep==1,]


if (nsim>2) {
    for (j in (2:nsim))
  {
    isaem[,2:7] <- isaem[,2:7]+disaem[disaem$rep==j,2:7]
  }
}

isaem[,2:7] <- 1/nsim*isaem[,2:7]
isaem[,9]<-NULL


isaemvr <- NULL
isaemvr <- disaemvr[disaemvr$rep==1,]


if (nsim>2) {
    for (j in (2:nsim))
  {
    isaemvr[,2:7] <- isaemvr[,2:7]+disaemvr[disaemvr$rep==j,2:7]
  }
}

isaemvr[,2:7] <- 1/nsim*isaemvr[,2:7]
isaemvr[,9]<-NULL

isaemvr$algo <- 'isaemvr'
isaemvr$rep <- NULL
isaemvr$iteration <- isaemvr$iteration*nbr/n




isaemsaga <- NULL
isaemsaga <- disaemsaga[disaemsaga$rep==1,]


if (nsim>2) {
    for (j in (2:nsim))
  {
    isaemsaga[,2:7] <- isaemsaga[,2:7]+disaemsaga[disaemsaga$rep==j,2:7]
  }
}

isaemsaga[,2:7] <- 1/nsim*isaemsaga[,2:7]
isaemsaga[,9]<-NULL

isaemsaga$algo <- 'isaemsaga'
isaemsaga$rep <- NULL
isaemsaga$iteration <- isaemsaga$iteration*nbr/n


em$algo <- 'EM'
saem$algo <- 'saem'
isaem$algo <- 'isaem'

em$rep <- NULL
saem$rep <- NULL
isaem$rep <- NULL


isaem$iteration <- isaem$iteration*nbr/n


variance <- rbind(isaem[,c(1,4,8)],
                  isaemsaga[,c(1,4,8)],
                  isaemvr[,c(1,4,8)],
                  em[,c(1,4,8)],
                  saem[,c(1,4,8)])

graphConvMC2_new(variance, title="",legend=TRUE)



# save.image("gmm.RData")
# write.csv(variance, file = "notebooks/singlerun.csv")



# ### PER EPOCH
# epochs = seq(1, K, by=n/nbr)
# em_ep <- em[1:(nbr*K/n),]
# em_ep$iteration <- 1:(nbr*K/n)
# saem_ep <- saem[1:(nbr*K/n),]
# saem_ep$iteration <- 1:(nbr*K/n)
# iem_ep <- iemseq[epochs,]
# iem_ep$iteration <- 1:(K/n)
# isaem_ep <- isaem[epochs,]
# isaem_ep$iteration <- 1:(K/n)


# start =1000
# end = K


# em_ep <- em
# em_ep$iteration <- n*em$iteration

# saem_ep <- saem
# saem_ep$iteration <- n*saem$iteration


# # variance <- rbind(iemseq[start:end,c(1,4,8)],
# #                   em_ep[2:length(epochs),c(1,4,8)],
# #                   saem_ep[2:length(epochs),c(1,4,8)],
# #                   isaem[start:end,c(1,4,8)])

# # graphConvMC2_new(variance, title="",legend=TRUE)



# start =1000
# end = K

# testiemseq <- iemseq
# testisaem <- isaem
# testem_ep <- em_ep
# testsaem_ep <- saem_ep


# testiemseq$iteration <- testiemseq$iteration/n
# testiemseq$algo <- 'IEM'
# testisaem$iteration <- testisaem$iteration/n
# testem_ep$iteration <- testem_ep$iteration/n
# testsaem_ep$iteration <- testsaem_ep$iteration/n



# variance <- rbind(testisaem[start:end,c(1,4,8)],
#                   testiemseq[start:end,c(1,4,8)],
#                   testem_ep[2:length(epochs),c(1,4,8)],testsaem_ep[2:length(epochs),c(1,4,8)])

# graphConvMC2_new(variance, title="",legend=TRUE)

