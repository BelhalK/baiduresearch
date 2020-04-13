compute.LLy<-function(z,xi, sample.digit,p, sigma,landmarks.p,landmarks.g,sigma.g,sigma.p) {
      ypred<-template.model(z, xi, p,landmarks.p,landmarks.g,sigma.g,sigma.p) #prediction of image
      sigma.cov <- diag(rep(sigma,p))
      inv.sigma<-solve(sigma.cov)
      DYF<-0.5*rowSums((sample.digit - ypred)*(sample.digit - ypred)%*%inv.sigma)
      U<-sum(DYF)
      return(U)
    }


compute.stat1<-function(yi,z, xi,p,kp,landmarks.p ,landmarks.g,sigma.g,sigma.p) {
  zi<-z
  temp  <- 0

  phi <- as.list(numeric(p^2))
  dim(phi) <- c(p,p)

  for (m in 1:p){
    for (j in 1:p){
      #Image Coordinate Standard
      x.ind = 2*m/p-1
      y.ind = 2*j/p-1
      rep.coord = matrix(c(x.ind,y.ind), nrow=1)
      coord <- t(apply(rep.coord, 2, rep, kg))
      
      #deformation computation
      kernel.deformation = exp(-(coord - landmarks.g)^2/(2*sigma.g**2))
      phi[[m,j]]= colSums(kernel.deformation)%*%t(zi)
      
      #template computation
      coord.template = rep.coord 
      rep.coord.template <- t(apply(coord.template, 2, rep, kp))
      kernel.template = exp(-(rep.coord.template - landmarks.p)^2/(2*sigma.p**2))

      template = colSums(kernel.template)*yi[m,j]
      temp <- temp + template
    }
  } 
  res = matrix(temp/p**2, nrow= kp)
  return(res)
}

compute.stat2<-function(z,xi,p,kp,landmarks.p,landmarks.g,sigma.g,sigma.p) { 
  zi<-z
  res.mat = matrix(NA,nrow=kp,ncol=2)

  phi <- as.list(numeric(p^2))
  dim(phi) <- c(p,p)


  temp <- 0
  for (m in 1:p){
    for (j in 1:p){
      #Image Coordinate Standard
      x.ind = 2*m/p-1
      y.ind = 2*j/p-1
      rep.coord = matrix(c(x.ind,y.ind), nrow=1)
      coord <- t(apply(rep.coord, 2, rep, kg))
      
      #deformation computation
      kernel.deformation = exp(-(coord - landmarks.g)^2/(2*sigma.g))
      phi[[m,j]]= colSums(kernel.deformation)%*%t(zi)

      #template computation
      coord.template = rep.coord - phi[[m,j]]
      rep.coord.template <- t(apply(coord.template, 2, rep, kp))
      kernel.template = exp(-(rep.coord.template - landmarks.p)^2/(2*sigma.p))

      
      template = colSums(kernel.template)%*%t(colSums(kernel.template))
      temp <- temp + template
    }
  } 

  res = temp/p**2
  return(res)
}

compute.stat3<-function(z) {
  res = t(z)%*%z
  return(res)
}

MCMC<-function(z, sample.digit, Gamma,xi, sigma,p,landmarks.p,landmarks.g ,nmcmc,sigma.g,sigma.p) {
  chol.omega<-try(chol(Gamma))
  somega<-solve(Gamma)
  
  U.z<-0.5*rowSums(z*(z%*%somega))
  U.y<-compute.LLy(z, xi, sample.digit,p, sigma,landmarks.p,landmarks.g,sigma.g,sigma.p)

  for(u in 1:nmcmc) { 
    zproposal<-matrix(rnorm(2*kg),ncol=kg)%*%chol.omega
    Uc.z<-0.5*rowSums(zproposal*(zproposal%*%somega))
    Uc.y<-compute.LLy(zproposal,xi, sample.digit,p, sigma,landmarks.p,landmarks.g,sigma.g,sigma.p)

    #MH acceptance ratio
    deltu<-Uc.y-U.y+Uc.z-U.z

    #accept reject step
    for (dim in 1:2){
      if (deltu[dim]<(-1)*log(runif(1))){
        z[dim,] = zproposal[dim,]
      }
    }
  }

  return(z)
}


