# GPL-3.0-or-later
# This file is part of MiniBatchSAEM
# Copyright (C) 2020, E. Kuhn, C. Matias, T. Rebafka
# 
# MiniBatchSAEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
# 
# MiniBatchSAEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with MiniBatchSAEM.  If not, see <https://www.gnu.org/licenses/>. 


# Fraility model
# Simulations for Figures 11 and 12 of the paper

library(ggplot2)
library(reshape2)
library(RColorBrewer)

source('frailty_miniSAEM.R') 

##******************************************************##
##**** set model parameters and other parameters *******##
##******************************************************##

I <- 5000
J <- 100
theta <- list(beta=c(2,3), sig2=2, lambda=3, rho=3.6)

set.seed(27)
data <- simul_frailty(I, J, theta) 

theta.init <- theta
theta.init$beta <- theta.init$beta + rnorm(length(theta.init$beta),0,2)
theta.init$lambda <- theta.init$lambda + rnorm(1,0,2)
theta.init$sig2 <- theta.init$sig2 + runif(1,0,2)

alpha_grid <- c(0.01,0.05,.1,.3,.5,.8,1)
REP <- 100
chauffe <- 2000
nb.run <- 6000
nb.it.max <- chauffe + nb.run
saveIt <- 1000 

##******************************************************##
##************ run the simulations *********************##
##******************************************************##

# ATTENTION: thi step is timeconsuming !

# for myseed in 1:28 run the following instructions in parallel 
# set.seed(myseed)
# nomficher <- paste('outputFrailtyLimit_',myseed,'.RData',sep='')
# 
# A <- length(alpha_grid)
# sol.alpha <- vector("list", REP)
# for (r in 1:REP){
#   sol.alpha[[r]] <- vector("list",A)
#   for (i in 1:A){
#     alpha <- alpha_grid[i]
#     cat('r=',r,'alpha=',alpha,'\n')
#     sol.alpha[[r]][[i]] <- saemMini(data, theta.init, alpha, nb.it.max, chauffe, saveIt)
#   }
#   save(sol.alpha,data,theta.init,alpha_grid, file=nomficher)
# }
# 
# ##******************************************************##
# ##*********** prepaere data for plots ******************##
# ##******************************************************##
# 
# itStar <- 8 # (corresponds to iteration 8000)
# 
# myseedVec <- 1:28
# for (myseed in myseedVec){
#   nomficher <- paste('outputFrailtyLimit_',myseed,'.RData',sep='')
#   load(nomficher)
# 
#   L <- sum(sapply(sol.alpha,length)>0)
#   lBeta1 <- lBeta2 <- lSig2 <- lLam <- matrix(NA,L,A)
#   for (l in 1:L){
#     lBeta1[l,] <- sapply(sol.alpha[[l]], function(x) x$beta.trace[itStar,1])
#     lBeta2[l,] <- sapply(sol.alpha[[l]], function(x) x$beta.trace[itStar,2])
#     lSig2[l,] <- sapply(sol.alpha[[l]], function(x) x$sig2.trace[itStar])
#     lLam[l,] <- sapply(sol.alpha[[l]], function(x) x$lam.trace[itStar])
#   }
# 
#   if (myseed==myseedVec[1]){
#     beta1 <- lBeta1
#     beta2 <- lBeta2
#     sig2 <- lSig2
#     lam <- lLam
#   }else{
#     beta1 <- rbind(beta1,lBeta1)
#     beta2 <- rbind(beta2,lBeta2)
#     sig2 <- rbind(sig2,lSig2)
#     lam <- rbind(lam,lLam)
#   }
# }
# 
# save(beta1, beta2, sig2, lam, file='outputFrailtyLimit_beta1.RData')

##******************************************************##
##**************** plot histogram **********************##
##******************************************************##

load('outputFrailtyLimit_beta1.RData')

df.beta1 <- data.frame(estim=c(beta1), alpha=rep(alpha_grid, each=nrow(beta1)))
df.beta1$alpha <- as.factor(df.beta1$alpha)
colPalete <- 'Set1'
colorList <- brewer.pal(n=7, name=colPalete)
ggplot(df.beta1) + geom_density(aes(x = estim, fill = alpha), alpha = .25) +
  scale_fill_manual(name="alpha", values=colorList, labels=as.character(alpha_grid))+
  xlab(' ') +
  theme(text = element_text(size=30))
# Figure 11 in the article

##******************************************************##
##**************** limit variance **********************##
##******************************************************##

# Fit limit variance to formula
lengthAlpha <- length(alpha_grid)
varBeta1 <- apply(beta1,2,var)
varBeta2 <- apply(beta2,2,var)/3*2
varSig2 <- apply(sig2,2,var)/10
varMatrix <- matrix(c(varBeta1, varBeta2, varSig2), lengthAlpha, 3) 
colnames(varMatrix) <- c('beta1','beta2','sig2')
df.varlim <- as.data.frame(varMatrix)
df.varlim$alpha <- alpha_grid
df.varlim <- melt(df.varlim, id='alpha')
names(df.varlim)[2:3] <- c('Param','estim')
colPalete <- 'Set1'  
colorList <- brewer.pal(n=8, name=colPalete)[c(1:5,7:8)]

g <- ggplot(df.varlim, aes(x=alpha)) +
  geom_line(aes(y = estim, colour = Param), size=1.2) +
  scale_colour_manual(values = colorList[1:6]) +
  theme(text = element_text(size=30)) +
  ylab('Limit variances')  +
  ylim(0,2e-7)  +   
  theme(axis.text.y = element_blank())

# add theoretical limit variance
grid <- seq(0.01,1,by=.01) 
G <- length(grid)
theovar <- matrix(NA,G,3)
Fcst <- 1.5
for (k in 1:2){
  sig02 <- sum((2-alpha_grid)/alpha_grid*varMatrix[,k])/sum(((2-alpha_grid)/alpha_grid)^2)
  theovar[,k] <-  Fcst*sig02*(2-grid)/grid
}
Fcst <- 1.9
k <- 3
sig02 <- sum((2-alpha_grid)/alpha_grid*varMatrix[,k])/sum(((2-alpha_grid)/alpha_grid)^2)
theovar[,k] <-  Fcst*sig02*(2-grid)/grid

theovar <- as.data.frame(theovar)
names(theovar) <- c('beta1','beta2','sig2')
theovar$alpha <- grid
theovar <- melt(theovar, id='alpha')
names(theovar)[2:3]  <-c('ParamTheo','theo')

g + geom_line(data=theovar, aes(y = theo, colour = ParamTheo), linetype="dashed", size=1.2) 

