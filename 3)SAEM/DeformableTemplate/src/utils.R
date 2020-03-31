compute.LLy<-function(z,xi, sample.digit,p, sigma,landmarks.p,landmarks.g) {
      ypred<-template.model(z, xi, p,landmarks.p,landmarks.g) #prediction of image

      sigma.cov <- diag(rep(sigma,p))
      inv.sigma<-solve(sigma.cov)
      
      DYF<-0.5*rowSums((sample.digit - ypred)*(sample.digit - ypred)%*%inv.sigma)
      U<-sum(DYF)

      return(U)
    }


compute.stat1<-function(yi,z, xi,p,kp,landmarks.p ,landmarks.g) {
  zi<-z
  temp  <- 0

  phi <- as.list(numeric(p^2))
  dim(phi) <- c(p,p)
  sigma.p = 1
  sigma.g = 1

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
      coord.template = rep.coord 
      rep.coord.template <- t(apply(coord.template, 2, rep, kp))
      kernel.template = exp(-(rep.coord.template - landmarks.p)^2/(2*sigma.p))

      template = colSums(kernel.template)*yi[m,j]
      temp <- temp + template
    }
  } 
  res = matrix(temp/p**2, nrow= kp)
  return(res)
}

compute.stat2<-function(z,xi,p,kp,landmarks.p,landmarks.g) { 
  zi<-z
  res.mat = matrix(NA,nrow=kp,ncol=2)

  phi <- as.list(numeric(p^2))
  dim(phi) <- c(p,p)
  sigma.p = 1
  sigma.g = 1

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

