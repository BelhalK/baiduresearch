################## Stochastic approximation - compute sufficient statistics (M-step) #####################
mstep.fi<-function(kiter, Uargs, Dargs, opt, structural.model, DYF, phiM, varList, phi, betas, suffStat,nb_replacement,indchosen,saemix.options,suffStat.vr,h.suffStat, indchosen.j,alphas,saemixObject) {
	# M-step - stochastic approximation
	# Input: kiter, Uargs, structural.model, DYF, phiM (unchanged)
	# Output: varList, phi, betas, suffStat (changed)
	#					mean.phi (created)
	# Update variances - TODO - check if here or elsewhere
	nb.etas<-length(varList$ind.eta)
	domega<-cutoff(mydiag(varList$omega[varList$ind.eta,varList$ind.eta]),.Machine$double.eps)
	omega.eta<-varList$omega[varList$ind.eta,varList$ind.eta,drop=FALSE]
	omega.eta<-omega.eta-mydiag(mydiag(varList$omega[varList$ind.eta,varList$ind.eta]))+mydiag(domega)
	#  print(varList$omega.eta)
	chol.omega<-try(chol(omega.eta))
	d1.omega<-Uargs$LCOV[,varList$ind.eta]%*%solve(omega.eta)
	d2.omega<-d1.omega%*%t(Uargs$LCOV[,varList$ind.eta])
	comega<-Uargs$COV2*d2.omega
	
	#old indiv stat
	stat1.indiv.old<-apply(phi[,varList$ind.eta,,drop=FALSE],c(1,2),sum) 
	stat2.indiv.old<-matrix(data=0,nrow=nb.etas,ncol=nb.etas)
	stat3.indiv.old<-apply(phi[,,]**2,c(1,2),sum) #  sum on phi**2, across 3rd dimension
    statr.indiv.old <- 0
    for (index in indchosen){
    	psiM<-transphi(alphas[[index]]$phiM,Dargs$transform.par)
		fpred<-structural.model(psiM, Dargs$IdM, Dargs$XM)
		ff.j<-matrix(fpred,nrow=Dargs$nobs,ncol=Uargs$nchains)
		for(k in 1:Uargs$nchains) phi[,,k]<-phiM[((k-1)*Dargs$N+1):(k*Dargs$N),]
    	phi.j <- alphas[[index]]$phi
	    mean.phi.j <- alphas[[index]]$mean.phi 
	    stat1.indiv.old[index,]<- apply(phi.j[index,varList$ind.eta,,drop=FALSE],c(1,2),sum) 
		stat3.indiv.old[index,]<-  phi.j[index,,]**2 #  sum on phi**2, across 3rd dimension
		for(k in 1:Uargs$nchains) {
			phik<-phi.j[,varList$ind.eta,k]
			stat2.indiv.old<-stat2.indiv.old+t(phik)%*%phik
			fk<-ff.j[index,k]
			if(!is.na(match(Dargs$error.model,c("constant","exponential"))))
				resk<-sum((Dargs$yobs[index]-fk)**2)
			statr.indiv.old<-statr.indiv.old+resk
		}
    }
	statr.indiv.old <- statr.indiv.old/Dargs$N

	#for hstats
	stat1.indiv.j<-apply(phi[,varList$ind.eta,,drop=FALSE],c(1,2),sum) 
	stat2.indiv.j<-matrix(data=0,nrow=nb.etas,ncol=nb.etas)
	stat3.indiv.j<-apply(phi[,,]**2,c(1,2),sum) #  sum on phi**2, across 3rd dimension
    statr.indiv.j <- 0
    for (index in indchosen.j){
    	psiM<-transphi(alphas[[index]]$phiM,Dargs$transform.par)
		fpred<-structural.model(psiM, Dargs$IdM, Dargs$XM)
		ff.j<-matrix(fpred,nrow=Dargs$nobs,ncol=Uargs$nchains)
		for(k in 1:Uargs$nchains) phi[,,k]<-phiM[((k-1)*Dargs$N+1):(k*Dargs$N),]
    	phi.j <- alphas[[index]]$phi
	    mean.phi.j <- alphas[[index]]$mean.phi 
	    stat1.indiv.j[index,]<- apply(phi.j[index,varList$ind.eta,,drop=FALSE],c(1,2),sum) 
		stat3.indiv.j[index,]<-  phi.j[index,,]**2 #  sum on phi**2, across 3rd dimension
		for(k in 1:Uargs$nchains) {
			phik<-phi.j[,varList$ind.eta,k]
			stat2.indiv.j<-stat2.indiv.j+t(phik)%*%phik
			fk<-ff.j[index,k]
			if(!is.na(match(Dargs$error.model,c("constant","exponential"))))
				resk<-sum((Dargs$yobs[index]-fk)**2)
			statr.indiv.j<-statr.indiv.j+resk
		}
    }
	statr.indiv.j <- statr.indiv.j/Dargs$N


 	#new indiv stat
	psiM<-transphi(phiM,Dargs$transform.par)
	fpred<-structural.model(psiM, Dargs$IdM, Dargs$XM)
	ff<-matrix(fpred,nrow=Dargs$nobs,ncol=Uargs$nchains)
	for(k in 1:Uargs$nchains) phi[,,k]<-phiM[((k-1)*Dargs$N+1):(k*Dargs$N),]

	stat1.indiv <- apply(phi[,varList$ind.eta,,drop=FALSE],c(1,2),sum)
	stat2.indiv<-matrix(data=0,nrow=nb.etas,ncol=nb.etas)
	stat3.indiv<-apply(phi[,,]**2,c(1,2),sum) #  sum on phi**2, across 3rd dimension
	statr.indiv<-0
	for(k in 1:Uargs$nchains) {
		phik<-phi[,varList$ind.eta,k]
		stat2.indiv<-stat2.indiv+t(phik)%*%phik
		fk<-ff[,k]
		if(!is.na(match(Dargs$error.model,c("constant","exponential"))))
			resk<-sum((Dargs$yobs-fk)**2) else {
				if(Dargs$error.model=="proportional")
					resk<-sum((Dargs$yobs-fk)**2/cutoff(fk**2,.Machine$double.eps)) else resk<-0
			}
		statr.indiv<-statr.indiv+resk
	}

	##update Vstats
	h.suffStat$h.stat1[indchosen,]  = h.suffStat$h.stat1[indchosen,] + (stat1.indiv[indchosen,] - stat1.indiv.old[indchosen,])
	h.suffStat$h.stat2  = h.suffStat$h.stat2 + (stat2.indiv - stat2.indiv.old)
	h.suffStat$h.stat3[indchosen,]  = h.suffStat$h.stat3[indchosen,] + (stat3.indiv[indchosen,] - stat3.indiv.old[indchosen,])
	h.suffStat$h.statr  = h.suffStat$h.statr + (statr.indiv - statr.indiv.old)

	vS1 = h.suffStat$h.stat1
	vS2 = h.suffStat$h.stat2
	vS3 = h.suffStat$h.stat3
	vSr = h.suffStat$h.statr

	#Variance Reduction Update
	rho = saemix.options$rho
	suffStat.vr$stat1.vr = (1 - rho)*suffStat.vr$stat1.vr + rho*vS1
	suffStat.vr$stat2.vr = (1 - rho)*suffStat.vr$stat2.vr + rho*vS2
	suffStat.vr$stat3.vr = (1 - rho)*suffStat.vr$stat3.vr + rho*vS3
	suffStat.vr$statr.vr = (1 - rho)*suffStat.vr$statr.vr + rho*vSr

	# Update sufficient statistics
	suffStat$statphi1<-suffStat$statphi1+opt$stepsize[kiter]*(suffStat.vr$stat1.vr/Uargs$nchains-suffStat$statphi1)
	suffStat$statphi2<-suffStat$statphi2+opt$stepsize[kiter]*(suffStat.vr$stat2.vr/Uargs$nchains-suffStat$statphi2)
	suffStat$statphi3<-suffStat$statphi3+opt$stepsize[kiter]*(suffStat.vr$stat3.vr/Uargs$nchains-suffStat$statphi3)
	suffStat$statrese<-suffStat$statrese+opt$stepsize[kiter]*(suffStat.vr$statr.vr/Uargs$nchains-suffStat$statrese)


	############# Maximisation
	##### fixed effects
	if (opt$flag.fmin && kiter>=opt$nbiter.sa) {
		temp<-d1.omega[Uargs$ind.fix11,]*(t(Uargs$COV1)%*%(suffStat$statphi1-Uargs$dstatCOV[,varList$ind.eta]))
		betas[Uargs$ind.fix11]<-solve(comega[Uargs$ind.fix11,Uargs$ind.fix11],rowSums(temp)) 
		# ECO TODO: utiliser optimise dans le cas de la dimension 1
		if(Dargs$type=="structural"){
			beta0<-optim(par=betas[Uargs$ind.fix10],fn=compute.Uy_c,phiM=phiM,pres=varList$pres,args=Uargs,Dargs=Dargs,DYF=DYF,control=list(maxit=opt$maxim.maxiter))$par # else
		} else {
			beta0<-optim(par=betas[Uargs$ind.fix10],fn=compute.Uy_d,phiM=phiM,args=Uargs,Dargs=Dargs,DYF=DYF,control=list(maxit=opt$maxim.maxiter))$par
		}
		betas[Uargs$ind.fix10]<-betas[Uargs$ind.fix10]+opt$stepsize[kiter]*(beta0-betas[Uargs$ind.fix10])
	} else {
		temp<-d1.omega[Uargs$ind.fix1,]*(t(Uargs$COV1)%*%(suffStat$statphi1-Uargs$dstatCOV[,varList$ind.eta]))
		betas[Uargs$ind.fix1]<-solve(comega[Uargs$ind.fix1,Uargs$ind.fix1],rowSums(temp)) 
	}
	
	varList$MCOV[Uargs$j.covariate]<-betas
	mean.phi<-Uargs$COV %*% varList$MCOV
	e1.phi<-mean.phi[,varList$ind.eta,drop=FALSE]
	

	# Covariance of the random effects
	omega.full<-matrix(data=0,nrow=Uargs$nb.parameters,ncol=Uargs$nb.parameters)
	omega.full[varList$ind.eta,varList$ind.eta]<-suffStat$statphi2/Dargs$N + t(e1.phi)%*%e1.phi/Dargs$N - t(suffStat$statphi1)%*%e1.phi/Dargs$N - t(e1.phi)%*%suffStat$statphi1/Dargs$N
	varList$omega[Uargs$indest.omega]<-omega.full[Uargs$indest.omega]
	
	# Simulated annealing (applied to the diagonal elements of omega)
	if (kiter<=opt$nbiter.sa) {
		diag.omega.full<-mydiag(omega.full)
		vec1<-diag.omega.full[Uargs$i1.omega2]
		vec2<-varList$diag.omega[Uargs$i1.omega2]*opt$alpha1.sa
		idx<-as.integer(vec1<vec2)
		varList$diag.omega[Uargs$i1.omega2]<-idx*vec2+(1-idx)*vec1
		varList$diag.omega[Uargs$i0.omega2]<-varList$diag.omega[Uargs$i0.omega2]*opt$alpha0.sa
	} else {
		varList$diag.omega<-mydiag(varList$omega)
	}
	varList$omega<-varList$omega-mydiag(mydiag(varList$omega))+mydiag(varList$diag.omega)
	
	# Residual error
	if (Dargs$error.model=="constant" | Dargs$error.model=="exponential") {
		sig2<-suffStat$statrese/Dargs$nobs
		varList$pres[1]<-sqrt(sig2)
	}
	if (Dargs$error.model=="proportional") {
		sig2<-suffStat$statrese/Dargs$nobs
		varList$pres[2]<-sqrt(sig2)
	}
	if (Dargs$error.model=="combined") {
		# ECO TODO: check and secure (when fpred<0 => NaN, & what happens if bres<0 ???)
		ABres<-optim(par=varList$pres,fn=ssq,y=Dargs$yM,f=fpred)$par
		if (kiter<=opt$nbiter.saemix[1]) {
			varList$pres[1]<-max(varList$pres[1]*opt$alpha1.sa,ABres[1])
			varList$pres[2]<-max(varList$pres[2]*opt$alpha1.sa,ABres[2])
		} else {
			varList$pres[1]<-varList$pres[1]+opt$stepsize[kiter]*(ABres[1]-varList$pres[1])
			varList$pres[2]<-varList$pres[2]+opt$stepsize[kiter]*(ABres[2]-varList$pres[2])
		}
	}

	#old indiv.j stat
	xmcmc.old.j<-estep.fi(kiter, Uargs, Dargs, opt, structural.model, DYF,saemixObject,indchosen.j, alphas)

	stat1.indiv.old.j<-apply(phi[,varList$ind.eta,,drop=FALSE],c(1,2),sum) 
	stat2.indiv.old.j<-matrix(data=0,nrow=nb.etas,ncol=nb.etas)
	stat3.indiv.old.j<-apply(phi[,,]**2,c(1,2),sum) #  sum on phi**2, across 3rd dimension
    statr.indiv.old.j <- 0
    for (index in indchosen.j){
    	psiM<-transphi(alphas[[index]]$phiM,Dargs$transform.par)
		fpred<-structural.model(psiM, Dargs$IdM, Dargs$XM)
		ff.j<-matrix(fpred,nrow=Dargs$nobs,ncol=Uargs$nchains)
		for(k in 1:Uargs$nchains) phi[,,k]<-phiM[((k-1)*Dargs$N+1):(k*Dargs$N),]
    	phi.j <- alphas[[index]]$phi
	    mean.phi.j <- alphas[[index]]$mean.phi 
	    stat1.indiv.old.j[index,]<- apply(phi.j[index,varList$ind.eta,,drop=FALSE],c(1,2),sum) 
		stat3.indiv.old.j[index,]<-  phi.j[index,,]**2 #  sum on phi**2, across 3rd dimension
		for(k in 1:Uargs$nchains) {
			phik<-phi.j[,varList$ind.eta,k]
			stat2.indiv.old.j<-stat2.indiv.old.j+t(phik)%*%phik
			fk<-ff.j[index,k]
			if(!is.na(match(Dargs$error.model,c("constant","exponential"))))
				resk<-sum((Dargs$yobs[index]-fk)**2)
			statr.indiv.old.j<-statr.indiv.old.j+resk
		}
    }
	statr.indiv.old.j <- statr.indiv.old.j/Dargs$N

    

    ##update hstats
	h.suffStat$h.stat1 = h.suffStat$h.stat1 + stat1.indiv.j - stat1.indiv.old.j
	h.suffStat$h.stat2 = h.suffStat$h.stat2 + stat2.indiv.j - stat2.indiv.old.j
	h.suffStat$h.stat3 = h.suffStat$h.stat3 + stat3.indiv.j - stat3.indiv.old.j
	h.suffStat$h.statr = h.suffStat$h.statr + statr.indiv.j - statr.indiv.old.j

	for (index in indchosen.j){
		alphas[[index]]$mean.phi <- mean.phi
	    alphas[[index]]$varList <- varList 
	    alphas[[index]]$phiM <- phiM 
	    alphas[[index]]$phi <- phi	
	}
    

	return(list(varList=varList,mean.phi=mean.phi,phi=phi,betas=betas,suffStat=suffStat,suffStat.vr= suffStat.vr,alphas=alphas))
}
