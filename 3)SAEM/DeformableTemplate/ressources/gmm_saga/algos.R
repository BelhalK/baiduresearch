mixt.simulate <-function(n,weight,mu,sigma)
{
  G <- length(mu)
  Z <- sample(1:G, n, prob=weight, replace=T)
  x<-NULL
  for (g in 1:G)
  {
    x<-c(x,rnorm(length(which(Z==g)),mu[g],sigma[g]))
  }
  return(x)
}



#-------------------------------------

mixt.saem <- function(x, theta0, K, K1=NULL, M=1, alpha=1)
{
  G<-length(mu)
  col.names <- c("iteration", paste0("p",1:G), paste0("mu",1:G), paste0("sigma",1:G))
  
  if (is.null(K1))  K1 <- 1
  K2 <- K - K1 #second phase iterations
  if (length(alpha)==1)
  gamma<-c(rep(1,K1),1/(1:K2)^alpha)
  else{
    L <- 10
    KL <- round(K2/L)
    alpha <- seq(alpha[1], alpha[2], length.out = L)
    gamma <- rep(1,K1)
    dl <- 0
    for (l in (1:L))
    {
      if (l==L)  KL <- K2 - (L-1)*KL
      gamma <- c(gamma,1/(dl + (1:KL))^alpha[l])
      dl <- (dl + KL)^(alpha[l]/alpha[l+1])
    }
  }
  theta.est <- matrix(NA,K+1,3*G+1)
  theta.est[1,] <- c(0, theta0$p, theta0$mu, theta0$sigma)
  
  
  theta<-theta0
  s<-list(s1=0,s2=0,s3=0)
  for (k in 1:K)
  {
    Z<-step.S(x,theta,M)
    s<-step.SA(x,Z,s,gamma[k])
    theta$mu<-step.M(s,n)
    theta.est[k+1,] <- c(k, theta0$p, theta$mu, theta0$sigma)
  }
  df <- as.data.frame(theta.est)
  names(df) <- col.names
  return(df)
}

mixt.em <- function(x, theta0, K)
{
  G<-length(mu)
  col.names <- c("iteration", paste0("p",1:G), paste0("mu",1:G), paste0("sigma",1:G))
  
  theta.est <- matrix(NA,K+1,3*G+1)
  theta.est[1,] <- c(0, theta0$p, theta0$mu, theta0$sigma)
  
  theta<-theta0
  for (k in 1:K)
  {
    # if (k %% n==0)
    # {
    #   print('EM')
    #   print(k)
    # }
    
    #Update the statistics
    s<-step.E(x,theta)

    #M-step
    theta$mu<-step.M(s,n)
    theta.est[k+1,] <- c(k, theta0$p, theta$mu, theta0$sigma)
  }
  
  df <- as.data.frame(theta.est)
  names(df) <- col.names
  return(df)
}



mixt.isaem <- function(x, theta0, K, K1=NULL, M=1, alpha=1,nbr)
{
  G<-length(mu)
  col.names <- c("iteration", paste0("p",1:G), paste0("mu",1:G), paste0("sigma",1:G))
  
  if (is.null(K1))  K1 <- 1
  K2 <- K - K1 #second phase iterations
  if (length(alpha)==1)
  gamma<-c(rep(1,K1),1/(1:K2)^alpha)
  else{
    L <- 10
    KL <- round(K2/L)
    alpha <- seq(alpha[1], alpha[2], length.out = L)
    gamma <- rep(1,K1)
    dl <- 0
    for (l in (1:L))
    {
      if (l==L)  KL <- K2 - (L-1)*KL
      gamma <- c(gamma,1/(dl + (1:KL))^alpha[l])
      dl <- (dl + KL)^(alpha[l]/alpha[l+1])
    }
  }
  theta.est <- matrix(NA,K+1,3*G+1)
  theta.est[1,] <- c(0, theta0$p, theta0$mu, theta0$sigma)
  
  
  theta<-theta0
  s<-list(s1=0,s2=0,s3=0)
  Z<-step.S(x,theta,M)
  n<-length(x)
  # l <- rep(1:n,K/n*nbr)
  # i <- 1:nbr
  for (k in 1:K)
  {
    i <- sample(1:n,nbr)
    # for (m in li){
    for (m in i){
      Z[m,,]<-step.S_replace(x[m],theta,M)
    }
    s<-step.SA(x,Z,s,gamma[k])
    theta$mu<-step.M(s,n)
    theta.est[k+1,] <- c(k, theta0$p, theta$mu, theta0$sigma)

    i <- i + nbr
  }
  df <- as.data.frame(theta.est)
  names(df) <- col.names
  return(df)
}


mixt.iem.seq <- function(x, theta0, K,nbr)
{
  G<-length(mu)
  col.names <- c("iteration", paste0("p",1:G), paste0("mu",1:G), paste0("sigma",1:G))
  
  theta.est <- matrix(NA,K+1,3*G+1)
  theta.est[1,] <- c(0, theta0$p, theta0$mu, theta0$sigma)
  tau <- compute.tau(x,theta0)
  theta<-theta0
  # tau.old <- compute.tau(x[1],theta0)
  s <- compute.stat(x,tau)
  l <- rep(1:n,K/n)
  i <- 1
  for (k in 1:K)
  {

    # if (k %% n==0)
    # {
    #   print('IEM')
    #   print(k)
    # }
    # i <- sample(1:n, 1)
    #Update the conditional expectation for the chosen datum
    oldtau <- tau[l[i],]
    tau[l[i],] <- compute.tau(x[l[i]],theta)
    
    #Update the statistics 
    s$s1 <- s$s1 + tau[l[i],] - oldtau
    s$s2 <- s$s2 + x[l[i]]*(tau[l[i],] - oldtau)
    # s <- compute.stat(x,tau)
    
    #M-step
    theta$mu<-step.M(s,n)
    theta.est[k+1,] <- c(k, theta0$p, theta$mu, theta0$sigma)

    i <- i +nbr
  }
  
  df <- as.data.frame(theta.est)
  names(df) <- col.names
  return(df)
}

mixt.isaemvr <- function(x, theta0, K, K1=NULL, M=1, alpha=1,nbr, rho)
{
  G<-length(mu)
  col.names <- c("iteration", paste0("p",1:G), paste0("mu",1:G), paste0("sigma",1:G))
  
  if (is.null(K1))  K1 <- 1
  K2 <- K - K1 #second phase iterations
  if (length(alpha)==1)
  gamma<-c(rep(1,K1),1/(1:K2)^alpha)
  # gamma<-c(1/(1:K1)^alpha,1/(K1:K2)^alpha)
  else{
    L <- 10
    KL <- round(K2/L)
    alpha <- seq(alpha[1], alpha[2], length.out = L)
    gamma <- rep(1,K1)
    dl <- 0
    for (l in (1:L))
    {
      if (l==L)  KL <- K2 - (L-1)*KL
      gamma <- c(gamma,1/(dl + (1:KL))^alpha[l])
      dl <- (dl + KL)^(alpha[l]/alpha[l+1])
    }
  }
  theta.est <- matrix(NA,K+1,3*G+1)
  theta.est[1,] <- c(0, theta0$p, theta0$mu, theta0$sigma)
  
  
  theta<-theta0
  s<-list(s1=0,s2=0,s3=0)
  Z<-step.S(x,theta,M)
  s <- stats <- s.e.0 <- compute.stat(x,Z)
  n<-length(x)
  l <- rep(1:n,K/n*nbr)
  i <- 1:nbr
  Z.e.0 <- Z
  for (k in 1:K)
  {
    # li <- sample(1:n,nbr)
    # for (m in li){
    for (m in l[i]){
      Z[m,,]<-step.S_replace(x[m],theta,M)
    }

    s.indiv.new <- 0
    s.indiv.e.0 <- 0
    Z.indiv.new <- 0
    Z.indiv.e.0 <- 0

    if (k%%(n/nbr) == 0)
    { 
      theta.e.0 <- theta
      Z.e.0<-step.S(x,theta.e.0,M)
      s.e.0 <- compute.stat(x,Z.e.0)
    }

    # i <- sample(1:n, 1)

    for (m in 1:M)
    {
      Z.m <- Z[l[i],,m]
      Z.m.e.0 <- Z.e.0[l[i],,m]
      Z.indiv.new <- Z.indiv.new + Z.m
      Z.indiv.e.0 <- Z.indiv.e.0 + Z.m.e.0
      s.indiv.new <- s.indiv.new + x[l[i]] %*% Z.m 
      s.indiv.e.0 <- s.indiv.e.0 + x[l[i]] %*% Z.m.e.0 
    }



    #Update statistics
    stats$s1 <- (1-rho)*stats$s1 + rho*((Z.indiv.new - Z.indiv.e.0)*n/M + s.e.0$s1)
    stats$s2 <- (1-rho)*stats$s2 + rho*((s.indiv.new - s.indiv.e.0)*n/M + s.e.0$s2)

    # stats$s1 <- (Z.indiv.new - Z.indiv.e.0)*n/M + s.e.0$s1
    # stats$s2 <- (s.indiv.new - s.indiv.e.0)*n/M + s.e.0$s2

    s$s1<-s$s1+gamma[k]*(stats$s1-s$s1)
    s$s2<-s$s2+gamma[k]*(stats$s2-s$s2)


    #M-step
    theta$mu<-step.M(s,n)
    theta.est[k+1,] <- c(k, theta0$p, theta$mu, theta0$sigma)

    i <- i + nbr
  }
  df <- as.data.frame(theta.est)
  names(df) <- col.names
  return(df)
}


mixt.isaemsaga <- function(x, theta0, K, K1=NULL, M=1, alpha=1,nbr, rho)
{
  G<-length(mu)
  col.names <- c("iteration", paste0("p",1:G), paste0("mu",1:G), paste0("sigma",1:G))
  
  if (is.null(K1))  K1 <- 1
  K2 <- K - K1 #second phase iterations
  if (length(alpha)==1)
  gamma<-c(rep(1,K1),1/(1:K2)^alpha)
  else{
    L <- 10
    KL <- round(K2/L)
    alpha <- seq(alpha[1], alpha[2], length.out = L)
    gamma <- rep(1,K1)
    dl <- 0
    for (l in (1:L))
    {
      if (l==L)  KL <- K2 - (L-1)*KL
      gamma <- c(gamma,1/(dl + (1:KL))^alpha[l])
      dl <- (dl + KL)^(alpha[l]/alpha[l+1])
    }
  }
  theta.est <- matrix(NA,K+1,3*G+1)
  theta.est[1,] <- c(0, theta0$p, theta0$mu, theta0$sigma)
  
  
  theta<-theta0
  s<-list(s1=0,s2=0,s3=0)
  Z<-step.S(x,theta,M)
  s <- stats <- v <- h <- compute.stat(x,Z)
  n<-length(x)
  li <- NULL
  alphas <- rep(list(theta0),n)

  li <- rep(sample(1:n,n), K/n)
  lj <- NULL
  for (index in 1:(K/n)){
    lj <- list.append(lj, sample(li[(1+(index-1)*n):(index*n)]))
  }
  i <- 1:nbr
  j <- 1:nbr

  oldZ <- h.Z <- h.oldZ <- Z
  for (k in 1:K)
  {

    for (m in li[i]){
      Z[m,,]<-step.S_replace(x[m],theta,M)
      oldZ[m,,]<-step.S_replace(x[m],alphas[[li[m]]],M)
    }

    newS.i <- 0
    oldS.i <- 0
    newZ.i <- 0
    oldZ.i <- 0

    for (m in 1:M)
    {
      Z.m <- Z[li[i],,m]
      oldZ.m <- oldZ[li[i],,m]
      newZ.i <- newZ.i + Z.m
      oldZ.i <- oldZ.i + oldZ.m

      newS.i <- newS.i + x[li[i]] %*% Z.m 
      oldS.i <- oldS.i + x[li[i]] %*% oldZ.m 
    }



    #Update statistics in a SAGA fashion
    v$s1 <- h$s1 + (newZ.i - oldZ.i)*n/M
    v$s2 <- h$s2 + (newS.i - oldS.i)*n/M

    stats$s1 <- (1-rho)*stats$s1 + rho*v$s1
    stats$s2 <- (1-rho)*stats$s2 + rho*v$s2


    #SA STEP
    s$s1<-s$s1+gamma[k]*(stats$s1-s$s1)
    s$s2<-s$s2+gamma[k]*(stats$s2-s$s2)
    

    #M-step
    oldtheta <- theta
    theta$mu<-step.M(s,n)
    theta.est[k+1,] <- c(k, theta0$p, theta$mu, theta0$sigma)

    oldalpha.j <- alphas[[lj[j]]]
    alphas[[lj[j]]] <- oldtheta

    for (m in lj[j]){
      h.Z[m,,]<-step.S_replace(x[m],oldtheta,M)
      h.oldZ[m,,]<-step.S_replace(x[m],oldalpha.j,M)
    }

    newS.j <- 0
    oldS.j <- 0
    newZ.j <- 0
    oldZ.j <- 0

    for (m in 1:M)
    {
      Z.m <- h.Z[lj[j],,m]
      oldZ.m <- h.oldZ[lj[j],,m]
      newZ.j <- newZ.j + Z.m
      oldZ.j <- oldZ.j + oldZ.m

      newS.j <- newS.j + x[lj[j]] %*% Z.m 
      oldS.j <- oldS.j + x[lj[j]] %*% oldZ.m 
    }

    h$s1 <- h$s1 + (newZ.j - oldZ.j)/M
    h$s2 <- h$s2 + (newS.j - oldS.j)/M

    i <- i+nbr
    j <- j+nbr
  }
  df <- as.data.frame(theta.est)
  names(df) <- col.names
  return(df)
}


# mixt.saga <- function(x, theta0, K,nbr, rho.saga)
# {
#    G<-length(mu)
#   col.names <- c("iteration", paste0("p",1:G), paste0("mu",1:G), paste0("sigma",1:G))
#   theta.est <- matrix(NA,K+1,3*G+1)
#   theta.est[1,] <- c(0, theta0$p, theta0$mu, theta0$sigma)
#   theta<-theta0
  
#   #Init
#   tau <- compute.tau(x,theta)
#   s <- v <- h <- compute.stat(x,tau)
#   n<-length(x)
#   li <- NULL
#   alphas <- rep(list(theta0),n)
#   # l <- sample(1:n,K,replace = TRUE)
#   # l <- rep(1:n,K/n)
#   li <- rep(sample(1:n,n), K/n)
#   lj <- NULL
#   for (index in 1:(K/n)){
#     lj <- list.append(lj, sample(li[(1+(index-1)*n):(index*n)]))
#   }
#   i <- 1:nbr
#   j <- 1:nbr
  
#   for (k in 1:K)
#   {
#     newtau.i<- compute.tau(x[li[i]],theta)
#     oldtau.i<- compute.tau(x[li[i]],alphas[[li[i]]])

#     v$s1 <- h$s1 + (newtau.i - oldtau.i)*n
#     v$s2 <- h$s2 + (x[li[i]]*newtau.i - x[li[i]]*oldtau.i)*n
    
#     s$s1 <- (1-rho.saga)*s$s1 + rho.saga*v$s1
#     s$s2 <- (1-rho.saga)*s$s2 + rho.saga*v$s2

#     oldtheta <- theta
#     theta$mu<-step.M(s,n)

#     theta.est[k+1,] <- c(k, theta0$p, theta$mu, theta0$sigma)

#     oldalpha.j <- alphas[[lj[j]]]
#     alphas[[lj[j]]] <- oldtheta
#     newtau.j<- compute.tau(x[lj[j]],oldtheta)
#     oldtau.j<- compute.tau(x[lj[j]],oldalpha.j)
#     # tau[lj[j],] <- newtau.i - oldtau.i
#     h$s1 <- h$s1 + (newtau.j - oldtau.j)
#     h$s2 <- h$s2 + (x[lj[j]]*newtau.j - x[lj[j]]*oldtau.j)

#     i <- i+nbr
#     j <- j+nbr
#   }
  
#   df <- as.data.frame(theta.est)
#   names(df) <- col.names
#   return(df)
# }





