#' Twotimescale SAEM: Fast Incremental variant
#'
#' This function uses algorithm SAEM to fit the logistic regression model with missing data.
#' @param X.obs Observed images matrix
#' @param maxruns Maximum number of iterations. The default is maxruns = 500.
#' @param tol_em The tolerance to stop SAEM. The default is tol_em = 1e-7.
#' @param nmcmc The MCMC length. The default is nmcmc = 2.
#' @param tau Rate \eqn{\tau}{\tau} in the step size \eqn{(k-k_{1})^{-\tau}}{(k-k1)^(-\tau)}. The default is tau = 1.
#' @param k1 Number of first iterations \eqn{k_{1}}{k1} in the step size \eqn{(k-k_{1})^{-\tau}}{(k-k1)^(-\tau)}. The default is k1=50.
#' @param seed An integer as a seed set for the radom generator. The default value is 200.
#' @return A list with components
#' \item{xi}{Estimated \eqn{\xi}{\xi}.}
#' \item{sigma}{Estimated \eqn{\Sigma}{\Sigma}.}
#' \item{Gamma}{Estiamated \eqn{\Gamma}{\Gamma}.}


fisaem <- function(X.obs,kp,kg,template.model,maxruns=500,tol_em=1e-7,
                      nmcmc=3,tau=1,k1=50, seed=200, print_iter=TRUE,
                       algo = "saem", batchsize=1, rho=0.5) {
    set.seed(seed)
  
    images <- as.matrix(X.obs)
    p=sqrt(nrow(images)) #dimension of the input
    n=ncol(images) #number of images in the dataset (n)

    sample.digit = matrix(images[,1], ncol=16,byrow=FALSE)  
    samples <-list(sample.digit,sample.digit)
    for (indiv in 1:n){
      samples[[indiv]] <-sample.digit #list of all images
    }

    cov.z.0 <- 1
    xi.0 <- 1
    sigma.0 <- 1

    cov <- diag(rep(cov.z.0,kg)) # covariance of the random effects
    Gamma <-list(cov,cov)
    for (k in 1:maxruns){
      Gamma[[k]] <-cov #list of all estimated cov of z
    }
  

    xi = matrix(xi.0,nrow=kp,ncol=(maxruns+1)) #template fixed parameters (1 X kp)
    sigma = matrix(sigma.0,nrow=1,ncol=(maxruns+1)) #residual errors variance

    theta0 <- list(xi = xi[,1], Gamma = Gamma[[1]], sigma=sigma[1] )
    alphas <- rep(list(theta0),n)

    # # Normal proposal covariance
    # omega.eta <- diag(rep(1,kg))
    # chol.omega<-try(chol(omega.eta))
    # somega<-solve(omega.eta)

    #random effects and proposal initialisation 
    chol.omega<-try(chol(Gamma[[1]]))
    z1 <- matrix(rnorm(2*kg),ncol=kg)%*%chol.omega
    z <- list(z1,z1) #random effects (2 X kg)

    #Individuals stats initialization
    S1.indiv = matrix(0,nrow=kp)
    S2.indiv = matrix(0,nrow=kp,ncol=kp)

    S1 <- list(S1.indiv,S1.indiv)
    S2 <- list(S2.indiv,S2.indiv)
    S3 <- list(Gamma[[1]],Gamma[[1]])

    for (indiv in 1:n){
      chol.omega<-try(chol(Gamma[[1]]))
      z1 <- matrix(rnorm(2*kg),ncol=kg)%*%chol.omega
      z[[indiv]] <- z1
      S1[[indiv]] <- S1.indiv
      S2[[indiv]] <- S2.indiv
      S3[[indiv]] <- Gamma[[1]]
    }
    S1.j <- S1.old <- S1.old.j <- S1
    S2.j <- S2.old <- S2.old.j <- S2
    S3.j <- S3.old <- S3.old.j <- S3

    #global stats
    suffStat <- hstats <- stats <- list(S1=0,S2=0,S3=0)


    zproposal <- z #initialise proposal random effects
    z.old <- z.j <- z.old.j <- z

    # landmarks
    landmarks.p = matrix(rnorm(2*kp),ncol=kp) #of template
    landmarks.g = matrix(rnorm(2*kg),ncol=kg) #of deformation
    print(maxruns)
    for (k in 1:maxruns) {
      print(k)
      #mini batch indices sampling
      index.i <- sample(1:n, batchsize)
      index.j <- sample(1:n, batchsize)
      while (index.j==index.i)
      { 
        index.j <- sample(1:n, batchsize)
      }

      #E-step
      for (indiv in index.i){
        z[[indiv]] <- MCMC(z[[indiv]], samples[[indiv]], Gamma[[k]],xi[,k], sigma[,k],p,landmarks.p,landmarks.g,nmcmc)
        z.old[[indiv]] <- MCMC(z.old[[indiv]], samples[[indiv]], alphas[[indiv]]$Gamma,alphas[[indiv]]$xi, alphas[[indiv]]$sigma,p,landmarks.p,landmarks.g,nmcmc)

        ### Compute individual statistics
        S1[[indiv]] = compute.stat1(samples[[indiv]],z[[indiv]], xi[,k], p, kp, landmarks.p, landmarks.g)
        S2[[indiv]] = compute.stat2(z[[indiv]],xi[,k], p, kp, landmarks.p,landmarks.g)
        S3[[indiv]] = compute.stat3(z[[indiv]])

        S1.old[[indiv]] = compute.stat1(samples[[indiv]],z.old[[indiv]], alphas[[indiv]]$xi, p, kp, landmarks.p, landmarks.g)
        S2.old[[indiv]] = compute.stat2(z.old[[indiv]],alphas[[indiv]]$xi, p, kp, landmarks.p,landmarks.g)
        S3.old[[indiv]] = compute.stat3(z.old[[indiv]])
      }

  
      if(k <k1){gamma <- 1}else{gamma <- 1/(k-(k1-1))^tau}
      #M-Step

      #Saga update 
      vS1 = hstats$S1 + (S1[[index.i]] - S1.old[[index.i]])
      vS2 = hstats$S2 + (S2[[index.i]] - S2.old[[index.i]])
      vS3 = hstats$S3 + (S3[[index.i]] - S3.old[[index.i]])

      stats$S1 = (1-rho)*stats$S1 + rho*vS1
      stats$S2 = (1-rho)*stats$S2 + rho*vS2
      stats$S3 = (1-rho)*stats$S3 + rho*vS3


      ###update sufficient statistics
      suffStat$S1 = suffStat$S1 + gamma*(Reduce("+",S1) - suffStat$S1)
      suffStat$S2 = suffStat$S2 + gamma*(Reduce("+",S2) - suffStat$S2)
      suffStat$S3 = suffStat$S3 + gamma*(Reduce("+",S3) - suffStat$S3)
      
      oldtheta <- list(xi = xi[,k], Gamma = Gamma[[k]], sigma=sigma[k] )

      ###update global parameters
      Gamma[[k+1]] = suffStat$S3/n
      xi[,k+1] = solve(suffStat$S2)%*%suffStat$S1
      sigma[,k+1] = (t(xi[,k+1])%*%suffStat$S2%*%xi[,k+1] - 2*xi[,k+1]%*%suffStat$S1)/(n*p**p)


      #saga like updates after global parms updates
      oldalpha.j <- alphas[[index.j]]
      alphas[[index.j]] <- oldtheta

      indiv = index.j
      for (indiv in index.j){
        z.j[[indiv]] <- MCMC(z.j[[indiv]], samples[[indiv]], oldtheta$Gamma,oldtheta$xi, oldtheta$sigma,p,landmarks.p,landmarks.g,nmcmc)
        z.old.j[[indiv]] <- MCMC(z.old.j[[indiv]], samples[[indiv]], oldalpha.j$Gamma,oldalpha.j$xi, oldalpha.j$sigma,p,landmarks.p,landmarks.g,nmcmc)

        ### Compute individual statistics
        S1.j[[indiv]] = compute.stat1(samples[[indiv]],z[[indiv]], oldtheta$xi, p, kp, landmarks.p, landmarks.g)
        S2.j[[indiv]] = compute.stat2(z[[indiv]],oldtheta$xi, p, kp, landmarks.p,landmarks.g)
        S3.j[[indiv]] = compute.stat3(z[[indiv]])

        S1.old.j[[indiv]] = compute.stat1(samples[[indiv]],z.old[[indiv]], oldalpha.j$xi, p, kp, landmarks.p, landmarks.g)
        S2.old.j[[indiv]] = compute.stat2(z.old[[indiv]],oldalpha.j$xi, p, kp, landmarks.p,landmarks.g)
        S3.old.j[[indiv]] = compute.stat3(z.old[[indiv]])
      }


      hstats$S1 <- hstats$S1 + (S1.j[[index.j]] - S1.old.j[[index.j]])
      hstats$S2 <- hstats$S2 + (S2.j[[index.j]] - S2.old.j[[index.j]])
      hstats$S3 <- hstats$S3 + (S3.j[[index.j]] - S3.old.j[[index.j]])

    }



  return(list(seqgamma=Gamma, seqxi = xi, seqsigma = sigma))
}
