#' Twotimescale SAEM: Variance Reduced
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


vrsaem <- function(X.obs,kp,kg,landmarks.p,landmarks.g, template.model,maxruns=500,tol_em=1e-7,
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

    S1 <- S.e.0.1 <- list(S1.indiv,S1.indiv)
    S2 <- S.e.0.2 <- list(S2.indiv,S2.indiv)
    S3 <- S.e.0.3 <- list(Gamma[[1]],Gamma[[1]])

    for (indiv in 1:n){
      chol.omega<-try(chol(Gamma[[1]]))
      z1 <- matrix(rnorm(2*kg),ncol=kg)%*%chol.omega
      z[[indiv]] <- z1
      S1[[indiv]] <- S.e.0.1[[indiv]] <- S1.indiv
      S2[[indiv]] <- S.e.0.2[[indiv]] <- S2.indiv
      S3[[indiv]] <- S.e.0.3[[indiv]] <- Gamma[[1]]
    }

    #global stats
    suffStat<- stats <- list(S1=0,S2=0,S3=0)


    zproposal <- z #initialise proposal random effects


    for (k in 1:maxruns) {
      print(k)
      #mini batch indices sampling
      index <- sample(1:n, batchsize)

      #E-step
      for (indiv in index){
        z[[indiv]] <- MCMC(z[[indiv]], samples[[indiv]], Gamma[[k]],xi[,k], sigma[,k],p,landmarks.p,landmarks.g,nmcmc)
        
        ### Compute individual statistics
        S1[[indiv]] = compute.stat1(samples[[indiv]],z[[indiv]], xi[,k], p, kp, landmarks.p, landmarks.g)
        S2[[indiv]] = compute.stat2(z[[indiv]],xi[,k], p, kp, landmarks.p,landmarks.g)
        S3[[indiv]] = compute.stat3(z[[indiv]])
      }


      if (k%%(n/batchsize) == 0)
      { 
        Gamma.e.0 <- Gamma[[k]]
        xi.e.0 <- xi[,k]
        sigma.e.0 <- sigma[,k]

        Z.e.0<-z
        Z.proposal.e.0<-zproposal

        S.e.0.1 <- S1
        S.e.0.2 <- S2
        S.e.0.3 <- S3
        for (indiv in 1:n){
          Z.e.0[[indiv]] <- MCMC(Z.e.0[[indiv]], samples[[indiv]], Gamma.e.0,xi.e.0, sigma.e.0,p,landmarks.p,landmarks.g,nmcmc)

          ### Compute individual statistics
          S.e.0.1[[indiv]] = compute.stat1(samples[[indiv]],Z.e.0[[indiv]], xi.e.0, p, kp, landmarks.p, landmarks.g)
          S.e.0.2[[indiv]] = compute.stat2(Z.e.0[[indiv]],xi.e.0, p, kp, landmarks.p,landmarks.g)
          S.e.0.3[[indiv]] = compute.stat3(Z.e.0[[indiv]])
        }
      }
      if(k <k1){gamma <- 1}else{gamma <- 1/(k-(k1-1))^tau}

      #M-Step
      ###update sufficient statistics
      stats$S1 = (1-rho)*stats$S1 + rho*((S1[[index]] - S.e.0.1[[index]]) + Reduce("+",S.e.0.1) )
      stats$S2 = (1-rho)*stats$S2 + rho*((S2[[index]] - S.e.0.2[[index]]) + Reduce("+",S.e.0.2) )
      stats$S3 = (1-rho)*stats$S3 + rho*((S3[[index]] - S.e.0.3[[index]]) + Reduce("+",S.e.0.3) )
      
      suffStat$S1 = suffStat$S1 + gamma*(stats$S1 - suffStat$S1)
      suffStat$S2 = suffStat$S2 + gamma*(stats$S2 - suffStat$S2)
      suffStat$S3 = suffStat$S3 + gamma*(stats$S3 - suffStat$S3)


      ###update global parameters
      Gamma[[k+1]] = suffStat$S3/n
      xi[,k+1] = solve(suffStat$S2)%*%suffStat$S1
      sigma[,k+1] = (t(xi[,k+1])%*%suffStat$S2%*%xi[,k+1] - 2*xi[,k+1]%*%suffStat$S1)/(n*p**p)
    }



  return(list(seqgamma=Gamma, seqxi = xi, seqsigma = sigma))
}
