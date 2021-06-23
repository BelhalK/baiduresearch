################## Stochastic approximation - compute sufficient statistics (M-step) #####################
mstep.fi<-function(kiter, Uargs, Dargs, opt, structural.model, DYF, phiM, varList, phi, betas, suffStat,nb_replacement,indchosen,saemix.options,suffStat.fi,h.suffStat, indchosen.j,phiMs,saemixObject) {
	# M-step - stochastic approximation
	# Input: kiter, Uargs, structural.model, DYF, phiM (unchanged)
	# Output: varList, phi, betas, suffStat (changed)
	#					mean.phi (created)
	# Update variances - TODO - check if here or elsewhere
	nb.etas<-length(varList$ind.eta)
	domega<-cutoff(mydiag(varList$omega[varList$ind.eta,varList$ind.eta]),.Machine$double.eps)
	omega.eta<-varList$omega[varList$ind.eta,varList$ind.eta,drop=FALSE]
	omega.eta<-omega.eta-mydiag(mydiag(varList$omega[varList$ind.eta,varList$ind.eta]))+mydiag(domega)
	
	chol.omega<-try(chol(omega.eta))
	d1.omega<-Uargs$LCOV[,varList$ind.eta]%*%solve(omega.eta)
	d2.omega<-d1.omega%*%t(Uargs$LCOV[,varList$ind.eta])
	comega<-Uargs$COV2*d2.omega
	
	
	phi.old.i <- phi.old.j <- phi.new.j <-phi
	phiM.old.i <- phiM.old.j<- phiM
	#new indiv stat
	psiM<-transphi(phiM,Dargs$transform.par)
	fpred<-structural.model(psiM, Dargs$IdM, Dargs$XM)
	if(Dargs$error.model=="exponential")
		fpred<-log(cutoff(fpred))
	ff<-matrix(fpred,nrow=Dargs$nobs,ncol=Uargs$nchains)
	for(k in 1:Uargs$nchains) phi[,,k]<-phiM[((k-1)*Dargs$N+1):(k*Dargs$N),]

	stat1.indiv <- apply(phi[indchosen,varList$ind.eta,,drop=FALSE],c(1,2),sum)
	stat2.indiv<-matrix(data=0,nrow=nb.etas,ncol=nb.etas)
	stat3.indiv<-apply(phi[indchosen,,]**2,c(1,2),sum) 
	statr.indiv<-0
	
	for(k in 1:Uargs$nchains) {
		phik<-phi[indchosen,varList$ind.eta,k]
		stat2.indiv<-stat2.indiv+t(phik)%*%phik
		fk<-ff[,k]
		if(!is.na(match(Dargs$error.model,c("constant","exponential"))))
			resk<-sum((Dargs$yobs[indchosen]-fk[indchosen])**2) 
		statr.indiv<-statr.indiv+resk
	}

	#new j hstats under current model bur for j indices
	stat1.indiv.new.j <- apply(phi[indchosen.j,varList$ind.eta,,drop=FALSE],c(1,2),sum)
	stat2.indiv.new.j<-matrix(data=0,nrow=nb.etas,ncol=nb.etas)
	stat3.indiv.new.j<-apply(phi[indchosen.j,,]**2,c(1,2),sum) 
	statr.indiv.new.j<-0
	
	for(k in 1:Uargs$nchains) {
		phik<-phi[indchosen.j,varList$ind.eta,k]
		stat2.indiv.new.j<-stat2.indiv.new.j+t(phik)%*%phik
		fk<-ff[,k]
		if(!is.na(match(Dargs$error.model,c("constant","exponential"))))
			resk<-sum((Dargs$yobs[indchosen.j]-fk[indchosen.j])**2)  
		statr.indiv.new.j<-statr.indiv.new.j+resk
	}

	#old indiv stat i
	for (index in indchosen){
      phiM.old.i[index,] <- phiMs[[index]]$phiM[index,]
    }
	psiM.old.i<-transphi(phiM.old.i,Dargs$transform.par)
	fpred<-structural.model(psiM.old.i, Dargs$IdM, Dargs$XM)
	if(Dargs$error.model=="exponential")
		fpred<-log(cutoff(fpred))
	ff<-matrix(fpred,nrow=Dargs$nobs,ncol=Uargs$nchains)
	for(k in 1:Uargs$nchains) phi.old.i[,,k]<-phiM.old.i[((k-1)*Dargs$N+1):(k*Dargs$N),]

	stat1.indiv.old <- apply(phi.old.i[indchosen,varList$ind.eta,,drop=FALSE],c(1,2),sum)
	stat2.indiv.old<-matrix(data=0,nrow=nb.etas,ncol=nb.etas)
	stat3.indiv.old<-apply(phi.old.i[indchosen,,]**2,c(1,2),sum) 
	statr.indiv.old<-0
	
	for(k in 1:Uargs$nchains) {
		phik<-phi.old.i[indchosen,varList$ind.eta,k]
		stat2.indiv.old<-stat2.indiv.old+t(phik)%*%phik
		fk<-ff[,k]
		if(!is.na(match(Dargs$error.model,c("constant","exponential"))))
			resk<-sum((Dargs$yobs[indchosen]-fk[indchosen])**2) 
		statr.indiv.old<-statr.indiv.old+resk
	}
 
	#update Vstats
	vS1 = h.suffStat$h.stat1
	vS2 = h.suffStat$h.stat2
	vS3 = h.suffStat$h.stat3
	vSr = h.suffStat$h.statr

	vS1[indchosen,]  = h.suffStat$h.stat1[indchosen,] + (stat1.indiv - stat1.indiv.old)
	vS2  = h.suffStat$h.stat2 + (stat2.indiv - stat2.indiv.old)
	vS3[indchosen,]  = h.suffStat$h.stat3[indchosen,] + (stat3.indiv - stat3.indiv.old)
	vSr  = h.suffStat$h.statr + (statr.indiv - statr.indiv.old)

	#Variance Reduction Update
	rho = saemix.options$rho
	suffStat.fi$stat1.fi = (1 - rho)*suffStat.fi$stat1.fi + rho*vS1
	suffStat.fi$stat2.fi = (1 - rho)*suffStat.fi$stat2.fi + rho*vS2
	suffStat.fi$stat3.fi = (1 - rho)*suffStat.fi$stat3.fi + rho*vS3
	suffStat.fi$statr.fi = (1 - rho)*suffStat.fi$statr.fi + rho*vSr

	# Update sufficient statistics
	suffStat$statphi1<-suffStat$statphi1+opt$stepsize[kiter]*(suffStat.fi$stat1.fi/Uargs$nchains-suffStat$statphi1)
	suffStat$statphi2<-suffStat$statphi2+opt$stepsize[kiter]*(suffStat.fi$stat2.fi/Uargs$nchains-suffStat$statphi2)
	suffStat$statphi3<-suffStat$statphi3+opt$stepsize[kiter]*(suffStat.fi$stat3.fi/Uargs$nchains-suffStat$statphi3)
	# suffStat$statrese<-suffStat$statrese+opt$stepsize[kiter]*(suffStat.fi$statr.fi/Uargs$nchains-suffStat$statrese)
	suffStat$statrese<-suffStat$statrese+opt$stepsize[kiter]*(statr.indiv/Uargs$nchains-suffStat$statrese)


	#old indv stat j for hstats
	for (index in indchosen.j){
      phiM.old.j[index,] <- phiMs[[index]]$phiM[index,]
    }
	psiM.old.j<-transphi(phiM.old.j,Dargs$transform.par)
	fpred<-structural.model(psiM.old.j, Dargs$IdM, Dargs$XM)
	if(Dargs$error.model=="exponential")
		fpred<-log(cutoff(fpred))
	ff<-matrix(fpred,nrow=Dargs$nobs,ncol=Uargs$nchains)
	for(k in 1:Uargs$nchains) phi.old.j[,,k]<-phiM.old.j[((k-1)*Dargs$N+1):(k*Dargs$N),]

	stat1.indiv.old.j <- apply(phi.old.j[indchosen.j,varList$ind.eta,,drop=FALSE],c(1,2),sum)
	stat2.indiv.old.j<-matrix(data=0,nrow=nb.etas,ncol=nb.etas)
	stat3.indiv.old.j<-apply(phi.old.j[indchosen.j,,]**2,c(1,2),sum) 
	statr.indiv.old.j<-0
	
	for(k in 1:Uargs$nchains) {
		phik<-phi.old.j[indchosen.j,varList$ind.eta,k]
		stat2.indiv.old.j<-stat2.indiv.old.j+t(phik)%*%phik
		fk<-ff[,k]
		if(!is.na(match(Dargs$error.model,c("constant","exponential"))))
			resk<-sum((Dargs$yobs[indchosen.j]-fk[indchosen.j])**2)
		statr.indiv.old.j<-statr.indiv.old.j+resk
	}
   

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


	##update hstats
	h.suffStat$h.stat1[indchosen.j,] = h.suffStat$h.stat1[indchosen.j,] + (stat1.indiv.new.j - stat1.indiv.old.j)/Dargs$N
	h.suffStat$h.stat2 = h.suffStat$h.stat2 + (stat2.indiv.new.j - stat2.indiv.old.j)/Dargs$N
	h.suffStat$h.stat3[indchosen.j,] = h.suffStat$h.stat3[indchosen.j,] + (stat3.indiv.new.j - stat3.indiv.old.j)/Dargs$N
	h.suffStat$h.statr = h.suffStat$h.statr + (statr.indiv.new.j - statr.indiv.old.j)/Dargs$N


	return(list(varList=varList,mean.phi=mean.phi,phi=phi,betas=betas,suffStat=suffStat,suffStat.fi= suffStat.fi,h.suffStat = h.suffStat,phiMs=phiMs))
}
