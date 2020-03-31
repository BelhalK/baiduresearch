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


saem <- function(X.obs,kp,kg,template.model,maxruns=500,tol_em=1e-7,
                      nmcmc=3,tau=1,k1=50, seed=200, print_iter=TRUE,
                       algo = "saem", batchsize=1) {
    set.seed(seed)
    
    images <- as.matrix(X.obs)
    p=sqrt(nrow(images)) #dimension of the input
    n=ncol(images) #number of images in the dataset (n)

    sample.digit = matrix(images[,1], ncol=16,byrow=FALSE)  
    samples <-list(sample.digit,sample.digit)
    for (indiv in 1:n){
      samples[[indiv]] <-sample.digit #list of all images
    }
    ptm <- Sys.time()

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

    #global stats
    suffStat<-list(S1=0,S2=0,S3=0)


    zproposal <- z #initialise proposal random effects
    # landmarks
    landmarks.p = matrix(rnorm(2*kp),ncol=kp) #of template
    landmarks.g = matrix(rnorm(2*kg),ncol=kg) #of deformation


    for (k in 1:maxruns) {

      #mini batch indices sampling
      if (algo == 'isaem'){
        index <- sample(1:n, batchsize)
      } else {
        index <- 1:n
      }

      #E-step
      for (indiv in index){
        cov <- Gamma[[k]]
        chol.omega<-try(chol(cov))
        somega<-solve(cov)
        
        U.z<-0.5*rowSums(z[[indiv]]*(z[[indiv]]%*%somega))
        U.y<-compute.LLy(z[[indiv]], xi[,k], samples[[indiv]],p, sigma[,k],landmarks.p,landmarks.g)

        for(u in 1:nmcmc) { 

          zproposal[[indiv]]<-matrix(rnorm(2*kg),ncol=kg)%*%chol.omega
          Uc.z<-0.5*rowSums(zproposal[[indiv]]*(zproposal[[indiv]]%*%somega))
          Uc.y<-compute.LLy(zproposal[[indiv]],xi[,k], samples[[indiv]],p, sigma[,k],landmarks.p,landmarks.g)

          #MH acceptance ratio
          deltu<-Uc.y-U.y+Uc.z-U.z
          #accept reject step
          for (dim in 1:2){
            if (deltu[dim]<(-1)*log(runif(1))){
              z[[indiv]][dim,] = zproposal[[indiv]][dim,]
            }
          }
        }

        #M-Step
        ### Compute individual and summed statistics
        S1[[indiv]] = compute.stat1(samples[[indiv]],z[[indiv]], xi[,k], p, kp, landmarks.p, landmarks.g)
        S2[[indiv]] = compute.stat2(z[[indiv]],xi[,k], p, kp, landmarks.p,landmarks.g)
        S3[[indiv]] = compute.stat3(z[[indiv]])
      }

      if(k <k1){gamma <- 1}else{gamma <- 1/(k-(k1-1))^tau}
      #M-Step
      ###update sufficient statistics
      suffStat$S1 = suffStat$S1 + gamma*(Reduce("+",S1) - suffStat$S1)
      suffStat$S2 = suffStat$S2 + gamma*(Reduce("+",S2) - suffStat$S2)
      suffStat$S3 = suffStat$S3 + gamma*(Reduce("+",S3) - suffStat$S3)
      
      ###update global parameters
      Gamma[[k]] = suffStat$S3/n
      xi[,k] = solve(suffStat$S2)%*%suffStat$S1
      sigma[,k] = (t(xi[,k])%*%suffStat$S2%*%xi[,k] - 2*xi[,k]%*%suffStat$S1)/(n*p**p)
    }



  return(list(seqgamma=Gamma, seqxi = xi, seqsigma = sigma))
}
