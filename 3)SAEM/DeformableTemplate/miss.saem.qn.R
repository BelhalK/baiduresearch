#' miss.saem with Quasi Newton for Incremental Method
#'
#' This function uses algorithm SAEM to fit the logistic regression model with missing data.
#' @param X.obs Design matrix with missingness \eqn{N \times p}{N * p}
#' @param y Response vector \eqn{N \times 1}{N * 1}
#' @param pos_var Index of selected covariates. The default is pos_var = 1:ncol(X.obs).
#' @param maxruns Maximum number of iterations. The default is maxruns = 500.
#' @param tol_em The tolerance to stop SAEM. The default is tol_em = 1e-7.
#' @param nmcmc The MCMC length. The default is nmcmc = 2.
#' @param tau Rate \eqn{\tau}{\tau} in the step size \eqn{(k-k_{1})^{-\tau}}{(k-k1)^(-\tau)}. The default is tau = 1.
#' @param k1 Number of first iterations \eqn{k_{1}}{k1} in the step size \eqn{(k-k_{1})^{-\tau}}{(k-k1)^(-\tau)}. The default is k1=50.
#' @param seed An integer as a seed set for the radom generator. The default value is 200.
#' @param print_iter If TRUE, miss.saem will print the estimated parameters in each iteration of SAEM.
#' @param var_cal If TRUE, miss.saem will calculate the variance of estimated parameters.
#' @param ll_obs_cal If TRUE, miss.saem will calculate the observed log-likelihood.
#' @return A list with components
#' \item{mu}{Estimated \eqn{\mu}{\mu}.}
#' \item{sig2}{Estimated \eqn{\Sigma}{\Sigma}.}
#' \item{beta}{Estiamated \eqn{\beta}{\beta}.}
#' \item{time_run}{Execution time.}
#' \item{seqbeta}{Sequence of \eqn{\beta}{\beta} estimated in each iteration.}
#' \item{seqbeta_avg}{Sequence of \eqn{\beta}{\beta} with averaging in each iteration.}
#' \item{ll}{Observed log-likelihood.}
#' \item{var_obs}{Estimated variance for estimated parameters.}
#' \item{std_obs}{Estimated standard error for estimated parameters.}
#' @import mvtnorm stats
#' @examples
#' # Generate dataset
#' N <- 100  # number of subjects
#' p <- 3     # number of explanatory variables
#' mu.star <- rep(0,p)  # mean of the explanatory variables
#' Sigma.star <- diag(rep(1,p)) # covariance
#' beta.star <- c(1, 1,  0) # coefficients
#' beta0.star <- 0 # intercept
#' beta.true = c(beta0.star,beta.star)
#' X.complete <- matrix(rnorm(N*p), nrow=N)%*%chol(Sigma.star) +
#'               matrix(rep(mu.star,N), nrow=N, byrow = TRUE)
#' p1 <- 1/(1+exp(-X.complete%*%beta.star-beta0.star))
#' y <- as.numeric(runif(N)<p1)

#' # Generate missingness
#' p.miss <- 0.10
#' patterns <- runif(N*p)<p.miss #missing completely at random
#' X.obs <- X.complete
#' X.obs[patterns] <- NA
#'
#' # SAEM
#' list.saem = miss.saem(X.obs,y)
#' print(list.saem$beta)
#' @export

miss.saem <- function(X.obs,kp,kg,template.model,maxruns=500,tol_em=1e-7,nmcmc=2,tau=1,k1=50, seed=200, print_iter=TRUE, var_cal=FALSE, ll_obs_cal=FALSE, algo = "saem", batchsize=1) {
    set.seed(seed)

    #judge
    images <- as.matrix(X.obs)
    p=sqrt(nrow(images)) #dimension of the input
    n=ncol(images) #number of images in the dataset (n)
    ptm <- Sys.time()
    
    cov.z.0 <- 1
    xi.0 <- 1
    sigma.0 <- 1

    cov <- diag(rep(cov.z.0,kg)) # covariance of the random effects
    Gamma <-cov
    for (k in 2:maxruns){
      Gamma <-list(Gamma, cov) #list of all estimated cov of z
    }
    xi = matrix(xi.0,nrow=kp,ncol=(maxruns+1)) #template fixed parameters (1 X kp)
    sigma = matrix(sigma.0,nrow=1,ncol=(maxruns+1))

    # # Normal proposal covariance
    # omega.eta <- diag(rep(1,kg))
    # chol.omega<-try(chol(omega.eta))
    # somega<-solve(omega.eta)

    #random effects and proposal initialisation 
    chol.omega.z<-try(chol(Gamma[[1]]))
    z1 <- matrix(rnorm(2*kg),ncol=kg)%*%chol.omega.z
    z <- z1 #random effects (2 X kg)
    for (indiv in 2:n){
      chol.omega.z<-try(chol(Gamma[[i]]))
      z1 <- matrix(rnorm(2*kg),ncol=kg)%*%chol.omega.z
      z <- list(z, z1)
    }
    zproposal <- z #initialise proposal random effects

    # landmarks
    landmarks.p = matrix(rnorm(2*kp),ncol=kp) #of template
    landmarks.g = matrix(rnorm(2*kg),ncol=kg) #of deformation

    suffStat<-list(S1=0,S2=0,S3=0)
    
    
    compute.LLy<-function(z,xi, sample.digit,p, sigma) {
      fpred<-template.model(z, xi, indiv, p)
      DYF<-0.5*((sample.digit-fpred)/sigma)**2+log(gpred)
      U<-colSums(DYF)
      return(U)
    }


    nmcmc = 3

    # while ((cstop>tol_em)*(k<maxruns)|(k<20)){
    for (k in 1:maxruns) {

      #E-step
      for (indiv in 1:n){
        sample.digit = matrix(images[,i], ncol=16,byrow=FALSE)  
        cov <- Gamma[[k]]
        chol.omega<-try(chol(cov))
        somega<-solve(cov)
        U.z<-0.5*rowSums(z[[indiv]]*(z[[indiv]]%*%somega))
        U.y<-compute.LLy(z[[indiv]], sample.digit,p, sigma)

        for(u in 1:nmcmc) { 
          zproposal[[indiv]]<-matrix(rnorm(2*kg),ncol=kg)%*%chol.omega
          Uc.z<-0.5*rowSums(zproposal[[indiv]]*(zproposal[[indiv]]%*%somega))
          Uc.y<-compute.LLy(zproposal[[indiv]],xi[,k], sample.digit,p, sigma)
          deltu<-Uc.y-U.y+Uc.z-U.z
          if (deltu<(-1)*log(runif(1))){
            z[[indiv]] = zproposal[[indiv]]
          }
        }
      }

      #M-Step
      ###update sufficient statistics
      suffStat$S1 = 
      suffStat$S2 = 
      suffStat$S3 = 

      ###update global parameters
      Gamma[[k]] = suffStat$S3/n
      xi[,k] = suffStat$S1/suffStat$S2
      sigma[1,k] = 
    }



  return(list(seqgamma=Gamma, seqxi = xi, seqsigma = sigma))
}
