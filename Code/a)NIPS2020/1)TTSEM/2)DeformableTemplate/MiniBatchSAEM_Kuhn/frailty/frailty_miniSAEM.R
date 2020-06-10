# GPL-3.0-or-later
# This file is part of MiniBatchSAEM
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

# 10/04/2020
# Frailty model - Mini-batch SAEM algorithm and data generation function

##******************************************************##
##**** data generation in the frailty model ************##
##******************************************************##

simul_frailty <- function(I, J, theta){
  X <- matrix(runif(I*J*length(theta$beta), 0, 1), I*J, length(theta$beta))
  b <- rnorm(I, 0, sqrt(theta$sig2))
  scale <- (theta$lambda*exp(X%*%theta$beta + rep(b,each=J)))^(-1/theta$rho)
  data <- rweibull(I*J, theta$rho, scale)    
  return(list(obs=data, b=b, I=I, J=J, X=X))
}

##******************************************************##
##**** minibatch SAEM algorithm - main function ********##
##******************************************************##

# saemMini <- function(data, theta.init, prop=.1, nb.it.max=60000, chauffe=50000, 
#                      saveIt = 1, nu=0.2, gamma.chauff=.6){
#   # saveIt: save results of every saveIt-th iteration
#   nbPbMax <- 5
#   eps <- 1e-3
#   
#   # Initialisation
#   theta <- theta.init
#   latentB <- data$b #rnorm(data$I, 0,2*sqrt(theta$sig2))  
#   s_lam <- exp(latentB)
#   s_sig <- theta$sig2 
#   
#   p <- length(theta.init$beta)
#   nbSaveIt <- floor(nb.it.max/saveIt)
#   sig2.trace <- lam.trace <- rep(NA, nbSaveIt)
#   beta.trace <- matrix(NA, nbSaveIt,p)
#   b.trace <- matrix(NA, nbSaveIt, data$I)
#   
#   sig2.it <- lam.it <- rho.it <- rep(NA, saveIt) # or nbSaveIt ???
#   beta.it <- matrix(NA, saveIt,p)
#   b.it <- matrix(NA, saveIt, data$I)
#   
#   countSaveIt <- currIt <- 1
#   
#   acceptreject <- 0
#   pb <- TRUE
#   count.pb <- 0
#   while(pb){
#     pb <- FALSE
#     for (nb.it in 1:nb.it.max){
#       multi <- rbinom(1, data$I, prop)
#       if (multi > 0) {
#         indices <- sample(1:data$I, multi, replace=F)
#         bCand <- rnorm(multi, latentB[indices], nu)
#         # simulation of latent variables
#         for (k in 1:multi) {
#           i <- indices[k]
#           b_cand <- bCand[k]
#           cst_i <- theta$lambda * sum((data$obs[((i-1)*data$J+1):(i*data$J)]) ^ theta$rho *
#                                         exp(data$X[((i-1)*data$J+1):(i*data$J), ] %*% theta$beta))
#           logratio <- -(exp(b_cand) - exp(latentB[i])) * cst_i + data$J*(b_cand - latentB[i]) + (latentB[i]^2 - b_cand^2) / 2 / theta$sig2
#           acceptreject <- log(runif(1)) < logratio
#           
#           if (!is.na(acceptreject)) {
#             if (acceptreject)
#               latentB[i] <- b_cand
#           } else{
#             break
#           }
#         }
#         if (is.na(acceptreject))
#           break
#       }
#       # Stochastic approximation
#       # step size parameter
#       gamma <- if (nb.it <= chauffe) gamma.chauff else (nb.it - chauffe)^-.6
#       
#       # update of sufficient statistics
#       s_sig <- (1 - gamma) * s_sig + gamma * mean(latentB^2)
#       s_lam <-  (1 - gamma) * s_lam + gamma * exp(latentB)
#       
#       # parameter update
#       theta$sig2 <- s_sig
#       res.beta <- newton.beta(theta, data, s_lam, nb.it.max = 3)
#       if (is.numeric(res.beta$crit)){
#         theta$beta <- res.beta$beta
#       }else{
#         pb <- TRUE
#         count.pb <- count.pb + 1
#         if (count.pb>=nbPbMax)
#           pb <- FALSE
#         cat("There's a convergence problem in beta! \n")
#         theta <- theta.init # choose new initial point
#         theta$beta <- theta$beta + rnorm(p, 0, count.pb)
#         theta$lambda <- rgamma(1,theta$lambda/count.pb, 1/count.pb)
#         theta$sig2 <- rgamma(1,theta$sig2/count.pb, 1/count.pb)
#         latentB <- rnorm(data$I, 0,2*sqrt(theta$sig2))  
#         s_lam <- exp(latentB)
#         s_sig <- theta$sig2 
#         break
#       }
#       theta$lambda <- max(c(data$I*data$J / sum(colSums(matrix(data$obs^theta$rho * exp(data$X %*% theta$beta),
#                                                                nrow = data$J, ncol = data$I)) * s_lam),eps))
#       sig2.it[currIt] <- theta$sig2
#       lam.it[currIt] <- theta$lambda
#       beta.it[currIt, ] <- theta$beta
#       b.it[currIt, ] <- latentB      
#       currIt <- currIt + 1
#       
#       if ((nb.it %% saveIt) == 0) {
#         sig2.trace[countSaveIt] <- mean(sig2.it)  # ?? klappt das mit NA ??
#         lam.trace[countSaveIt] <- mean(lam.it) 
#         beta.trace[countSaveIt, ] <- colMeans(beta.it)
#         b.trace[countSaveIt, ] <- colMeans(b.it)
#         countSaveIt <- countSaveIt + 1
#         currIt <- 1
#       }
#     }
#   }
#   return(list(sig2.trace=sig2.trace, lam.trace=lam.trace, 
#               beta.trace=beta.trace, b.trace=b.trace)) 
# }

saemMini <- function(data, theta.init, prop=.1, nb.it.max=60000, chauffe=50000, 
                     nu=0.2, gamma.chauff=.6){
  nbPbMax <- 5 # limit convergence problems in Newton method updating beta
  eps <- 1e-3
  
  # Initialisation
  theta <- theta.init
  latentB <- rnorm(data$I, 0,2*sqrt(theta$sig2))  # data$b
  s_lam <- exp(latentB)
  s_sig <- theta$sig2 
  
  p <- length(theta.init$beta)
  sig2.trace <- lam.trace <- rep(NA, nb.it.max)
  beta.trace <- matrix(NA, nb.it.max, p)
  b.trace <- matrix(NA, nb.it.max, data$I)
  
  currIt <- 1
  acceptreject <- 0
  pb <- TRUE
  count.pb <- 0
  while(pb){
    pb <- FALSE
    for (nb.it in 1:nb.it.max){
      multi <- rbinom(1, data$I, prop)
      if (multi > 0) {
        indices <- sample(1:data$I, multi, replace=F)
        bCand <- rnorm(multi, latentB[indices], nu)
        # simulation of latent variables
        for (k in 1:multi) {
          i <- indices[k]
          b_cand <- bCand[k]
          cst_i <- theta$lambda * sum((data$obs[((i-1)*data$J+1):(i*data$J)]) ^ theta$rho *
                  exp(data$X[((i-1)*data$J+1):(i*data$J), ] %*% theta$beta))
          logratio <- -(exp(b_cand) - exp(latentB[i])) * cst_i + data$J*(b_cand - latentB[i]) + (latentB[i]^2 - b_cand^2) / 2 / theta$sig2
          acceptreject <- log(runif(1)) < logratio
          
          if (!is.na(acceptreject)) {
            if (acceptreject)
              latentB[i] <- b_cand
          } else{
            break
          }
        }
        if (is.na(acceptreject))
          break
      }
      # Stochastic approximation
      # step size parameter
      gamma <- if (nb.it <= chauffe) gamma.chauff else (nb.it - chauffe)^-.6
      
      # update of sufficient statistics
      s_sig <- (1 - gamma) * s_sig + gamma * mean(latentB^2)
      s_lam <-  (1 - gamma) * s_lam + gamma * exp(latentB)
      
      # parameter update
      theta$sig2 <- s_sig
      res.beta <- newton.beta(theta, data, s_lam, nb.it.max = 3)
      if (is.numeric(res.beta$crit)){
        theta$beta <- res.beta$beta
      }else{
        pb <- TRUE
        count.pb <- count.pb + 1
        if (count.pb>=nbPbMax)
          pb <- FALSE
        cat("There's a convergence problem in beta! \n")
        theta <- theta.init # choose new initial point
        theta$beta <- theta$beta + rnorm(p, 0, count.pb)
        theta$lambda <- rgamma(1,theta$lambda/count.pb, 1/count.pb)
        theta$sig2 <- rgamma(1,theta$sig2/count.pb, 1/count.pb)
        latentB <- rnorm(data$I, 0,2*sqrt(theta$sig2))  
        s_lam <- exp(latentB)
        s_sig <- theta$sig2 
        break
      }
      theta$lambda <- max(c(data$I*data$J / sum(colSums(matrix(data$obs^theta$rho * exp(data$X %*% theta$beta),
              nrow = data$J, ncol = data$I)) * s_lam),eps))
      sig2.trace[currIt] <- theta$sig2
      lam.trace[currIt] <- theta$lambda
      beta.trace[currIt, ] <- theta$beta
      b.trace[currIt, ] <- latentB      
      currIt <- currIt + 1
    }
  }
  return(list(sig2.trace=sig2.trace, lam.trace=lam.trace, 
              beta.trace=beta.trace, b.trace=b.trace)) 
}

##******************************************************##
##*********** auxiliary functions for SAEM *************##
##******************************************************##


# Newton method for estimation of parameter rho
newton.rho <- function(theta, data, s_lam, nb.it.max=10, eps=1e-6){
  not.converged <- T
  rho <- theta$rho
  nb.it <- 0
  
  logTij <- log(data$obs)
  sumlogTij <- sum(logTij)
  Kij <- logTij*exp(data$X%*%theta$beta)*rep(s_lam,each=data$J)  # IJ - vector
  
  while(not.converged & (nb.it < nb.it.max)){
    nb.it <- nb.it + 1
    Mijrho <- data$obs^rho*Kij
    derivee <- data$I*data$J/rho+sumlogTij-sum(Mijrho)*theta$lambda
    derivee2 <- -data$I*data$J/rho^2 - sum(Mijrho*logTij)*theta$lambda
    rho <- rho - derivee/derivee2
    not.converged <- (abs(derivee)>eps)
    
    if (is.na(not.converged)){ 
      not.converged <- FALSE
    }else{
      if (rho<1) 
        rho <- 1 + eps
    }
  }
  
  return(list(rho=rho, crit = nb.it))
}

# Newton method for estimation of parameter beta
newton.beta <- function(theta, data, s_lam, nb.it.max=10, eps=1e-6){
  not.converged <- T
  beta <- theta$beta
  p <- length(beta)
  nb.it <- 0
  
  C <- colSums(data$X)/theta$lambda  # p-vector
  Pij <- data$X*matrix(data$obs^theta$rho*rep(s_lam,each=data$J),nrow=data$I*data$J,ncol=p) # IJ x p
  
  while(not.converged & (nb.it < nb.it.max)){
    nb.it <- nb.it + 1
    Qijbeta <- Pij* matrix(exp(data$X%*%beta),nrow=data$I*data$J,ncol=p) # IJ x p
    derivee <- C - colSums(Qijbeta)
    derivee2 <-  -t(data$X)%*%Qijbeta  
    
    trySolve <- try(solve(derivee2,derivee))
    if (is.numeric(trySolve)){
      beta <- beta - trySolve
    }else{
      crit = "Problem in solve"
      break
    }
    not.converged <- (max(abs(derivee))>eps)
    if (is.na(not.converged)) not.converged <- FALSE
    crit <- nb.it
  }
  return(list(beta=beta, crit = crit))
}

