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
# Simulations for Figures 9 and 10 of the paper

library(ggplot2)
library(reshape2)
library(RColorBrewer)

source('frailty_miniSAEM.R')

##******************************************************##
##**** set model parameters and other parameters *******##
##******************************************************##

I <- 5000
J <- 100
theta <- theta.init <- list(beta=c(2,3), sig2=2, lambda=3, rho=3.6)
alpha_grid <- c(0.01,0.05,.1,.3,.5,.8,1)  
chauffe_batch <- 20
nb.run.alpha_batch <- 20
nb.it.max_batch <- chauffe_batch + nb.run.alpha_batch

##******************************************************##
##************ run the simulations *********************##
##******************************************************##

# ATTENTION: the next step is timeconsuming !

# # we ran the following code for myseed in 1:13 on several servers at the same time (with different seeds and different values for REP)
# REP <- 200
# myseed <- 1 
# 
# nomficher <- paste('output19dec2018_',myseed,'.RData',sep='')
# set.seed(myseed)
# A <- length(alpha_grid)
# sol.alpha <- vector("list", REP)
# for (r in 1:REP){
#   data <- simul_frailty(I, J, theta)
#   sol.alpha[[r]] <- vector("list", A)
#   theta.init$beta <- theta.init$beta + rnorm(length(theta.init$beta), 0, 2)
#   theta.init$lambda <- theta.init$lambda + rnorm(1, 0, 2)
#   for (i in 1:A){
#     alpha <- alpha_grid[i]
#     cat('r=',r,'alpha=',alpha,'\n')
#     chauffe <- round(chauffe_batch/alpha)
#     nb.run <- round(nb.run.alpha_batch/alpha)
#     nb.it.max <- chauffe + nb.run
#     sol.alpha[[r]][[i]] <- saemMini(data, theta.init, alpha, nb.it.max, chauffe, saveIt=1)
#   }
#   save(sol.alpha, data, theta.init, alpha_grid, file=nomficher)
# }

##******************************************************##
##***** arrange simulation results by epochs************##
##******************************************************##

# cleanNARows <- function(mat){
#   mat <- mat[rowSums(is.na(mat))==0,]
#   return(mat)
# }
# 
# it.batchVec <- 1:30  # vector of epochs to study
# myseedVec <- 1:13  
# S <- length(myseedVec)
# 
# for (it.batch in it.batchVec){
#   cat('it.batch',it.batch,'\n')
#   nomfichier <- paste('resItBatch',it.batch,'.RData',sep='')
#   for (s in 1:S){
#     # load data
#     myseed <- myseedVec[s]
#     nomficherData <- paste('output19dec2018_',myseed,'.RData',sep='')
#     load(nomficherData)
#     
#     L <- sum(sapply(sol.alpha,length)>0)
#     if (L<length(sol.alpha))
#       L <- L-1
#     
#     # extract data corresponding to iteration number it.batch
#     sig2.alpha <- lam.alpha <- beta1.alpha <- beta2.alpha <- matrix(NA,L,A)
#     
#     for (j in 1:A){
#       alpha <- alpha_grid[j]
#       it.j <- round(it.batch/alpha)
#       for (l in 1:L){
#         sig2.alpha[l,j] <- sol.alpha[[l]][[j]]$sig2.trace[it.j]
#         lam.alpha[l,j] <- sol.alpha[[l]][[j]]$lam.trace[it.j]
#         beta1.alpha[l,j] <- sol.alpha[[l]][[j]]$beta.trace[it.j,1]
#         beta2.alpha[l,j] <- sol.alpha[[l]][[j]]$beta.trace[it.j,2]
#       }
#     }
#     
#     if (myseed!=1){
#       sig2.alpha <- cleanNARows(sig2.alpha)
#       lam.alpha <- cleanNARows(lam.alpha)
#       beta1.alpha <- cleanNARows(beta1.alpha)
#       beta2.alpha <- cleanNARows(beta2.alpha)
#       
#       load(nomfichier)
#       
#       sig2 <- rbind(sig2,sig2.alpha)
#       lam <- rbind(lam,lam.alpha)
#       beta1 <- rbind(beta1,beta1.alpha)
#       beta2 <- rbind(beta2,beta2.alpha)
#       
#       save(sig2, lam, beta1, beta2, file=nomfichier)
#     }else{
#       sig2 <- cleanNARows(sig2.alpha)
#       lam <- cleanNARows(lam.alpha)
#       beta1 <- cleanNARows(beta1.alpha)
#       beta2 <- cleanNARows(beta2.alpha)
#       save(sig2, lam, beta1, beta2, file=nomfichier)
#     }
#     rm(sol.alpha)
#   }
# }

##******************************************************##
##*********** prepaere data for plots ******************##
##******************************************************##

# summStat <- function(mat, param, alpha_grid){
#   moy <- colMeans(mat, na.rm=T)
#   variance <- apply(mat,2, function(x) var(x,na.rm=T))
#   mse <- (moy-param)^2 + variance
#   stat <- rbind(moy,variance,mse)
#   colnames(stat) <- as.numeric(alpha_grid)
#   return(stat)
# }
# 
# it.batchVec <- 1:30  # vector of epochs to study 
# I <- length(it.batchVec)
# A <- length(alpha_grid)
# resMSEbeta1 <- matrix(NA,I,A)
# confint95beta1 <- confint05beta1  <- matrix(NA,I,A)
# 
# for (i in 1:I){
#   it.batch <- it.batchVec[i]
#   nomfichier <- paste('resItBatch',it.batch,'.RData',sep='')
#   load(nomfichier)
#   resMSEbeta1[i,] <- summStat(beta1,theta$beta[1],alpha_grid)[3,]
#   g <- .1
#   confint95beta1[i,] <- apply(beta1, 2, function(x) quantile(x, 1-g))
#   confint05beta1[i,] <- apply(beta1, 2, function(x) quantile(x, g))
# }
# 
# save(resMSEbeta1, confint95beta1, confint05beta1, file='outputFrailtyAccel_beta1.RData')

##******************************************************##
##****************** draw figure 10 ********************##
##******************************************************##

load('outputFrailtyAccel_beta1.RData')

colnames(resMSEbeta1) <- alpha_grid
logMSE <- as.data.frame(log(resMSEbeta1))
logMSE$nbIter <- 1:nrow(logMSE) 
longLogMSE <- melt(logMSE, id='nbIter')
names(longLogMSE)[2] <- 'alpha'
names(longLogMSE)[3] <- 'logMSE'

colPalete <- 'Set1'  
colorList <- brewer.pal(n=9, name=colPalete)[c(1:5,8:9)]#[c(1:5,7:8)]
A <- length(alpha_grid)
colInd <- colorList[rep(1:A,each=nrow(logMSE))]

ggplot(data=longLogMSE, aes(x=nbIter, y=logMSE, colour=alpha, linetype=alpha)) +
  geom_line(size=1.2) +
  xlab('Number of epochs') + ylab("log(MSE)")  +
  scale_colour_manual(labels = as.character(alpha_grid), values = colorList) +
  theme(text = element_text(size=30))
# Figure 10 of the paper

##******************************************************##
##****************** draw figure 9 *********************##
##******************************************************##

colnames(confint95beta1) <- alpha_grid
colnames(confint05beta1) <- alpha_grid
confint95beta1 <- as.data.frame(confint95beta1)
confint05beta1 <- as.data.frame(confint05beta1)
confint95beta1$nbIter <- 1:nrow(confint95beta1) 
confint05beta1$nbIter <- 1:nrow(confint05beta1)
longConfint95beta1 <- melt(confint95beta1, id='nbIter')
longConfint05beta1 <- melt(confint05beta1, id='nbIter')
names(longConfint95beta1)[2] <- names(longConfint05beta1)[2] <- 'alpha'
names(longConfint95beta1)[3] <- 'CI95'
names(longConfint05beta1)[3] <- 'CI05'

longConfintbeta1 <- longConfint95beta1
longConfintbeta1$CI05 <- longConfint05beta1$CI05

dfPart <- longConfintbeta1[1:150,]
ggplot(dfPart, aes(x=nbIter)) +
  geom_line(aes(y = CI95, colour = alpha, linetype=alpha), size=1.2) +
  geom_line(aes(y = CI05, colour = alpha, linetype=alpha),size=1.2) +
  scale_colour_manual(labels = as.character(alpha_grid[1:5]), values = colorList[1:5]) +
  theme(text = element_text(size=30)) +
  ylab('beta1') + xlab('Number of epochs')
# Figure 9 of the paper

