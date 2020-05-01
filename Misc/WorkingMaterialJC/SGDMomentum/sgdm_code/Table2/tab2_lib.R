library(e1071)

expit <- function(x) sapply(x, function(i) { if(i > 60) return(1); if(i < -60) return(0); return(exp(i)/(1 + exp(i)))})

par_rmvnorm <- function(N=1e5,p=1e3) {
  Sigma_X = toeplitz(0.5^abs(1-1:p))
  draw_X <- function(i) {
    i=1
    return(rmvnorm(1e3, mean=rep(0,p), sigma=Sigma_X))
  }
  X = mclapply(1:100, function(i) draw_X(i))
  return(X)
}

du_expit <- function(x) sapply(x, function(i) { if(i > 60) return(1); if(i < -60) return(0); return(exp(-i)/((1 + exp(-i))^2))})
L2 = function(x, y) sqrt(sum((x-y)^2))
L2.p = function(x, y) {
  sqrt(sum((x-y)^2)) / length(x) 
}

fast_sqrtm <- function(M) {
  p = ncol(M)
  I = diag(p)
  U = matrix(1, nrow=p, ncol=p)
  # Check for equicorr matrix.
  equicor_obj <- function(par) {
    Mhat = par[1] * diag(p) + par[2]* U
    sum((Mhat - M)^2)
  }
  out = optim(par=c(0, 0), fn = equicor_obj, 
              lower = c(-100, -100), upper=c(100, 100),
              method="L-BFGS-B")
  ## special case
  if(all(M == I)) {
    out$par = c(1, 0)
    out$value = 0
  }
  if(out$value < 1e-6) {
    ## This is indeed equicor
    c1 = out$par[1]
    c2 = out$par[2]
    a = sqrt(c1)
    all_b = polyroot(c(-c2 * a, 2 * c1, a * p))
    b = Re(all_b[1])
    if(all(eigen(a * I + all_b[2] * U)$values > 0)) {
      b = all_b[2]
    }
    return(a * I + b * U)
  }
  
}

## Compact code for implicit SGD.
##  example:
##    d = gen_data(model_name="logistic", N=1e4)
##    out = implicit_sgd(d, verbose=T)
##
gen_data_star <- function(theta_star, X, model_name="gaussian", sigma_noise=1) {
  # unload experimental params here. TODO: extend to match our current experimental setup.
  # theta_star = (-1)^seq(1, p) * thetaStar_coeff * exp(-0.75 * seq(1, p))
  ##
  N = nrow(X)
  p = ncol(X)
  pred = X %*% theta_star  # predictor.
  # 
  glm_link = NA # glm link function. 
  Y = NA
  #
  if(model_name == "gaussian") {
    glm_link = function(x) x
    du_glm_link = function(x) 1
    noise = rnorm(N, sd=sigma_noise)
    Y = glm_link(pred) + noise
  } else if(model_name == "poisson") {
    glm_link = function(x) exp(x)
    du_glm_link = function(x) exp(x)
    Y = rpois(N, lambda=glm_link(pred))
  } else if(model_name=="binomial") {
    glm_link = function(x) expit(x)
    du_glm_link = function(x) du_expit(x)
    Y = rbinom(N, size=1, prob=glm_link(pred))
  } else {
    stop("model not implemented.")
  }
  
  return(list(model=model_name, X=X, Y=Y, glm_link=glm_link, 
              du_glm_link=du_glm_link, theta_star=theta_star))
}

gen_data <- function(model_name="gaussian", N=1000, p=20, 
                     sigma_x="id", rho=.15, theta_coeff=1,
                     sigma_noise=1, true_param="classic",
                     sigma_sqrt=NA) {
  # unload experimental params here. TODO: extend to match our current experimental setup.
  
  # classic theta star
  if(true_param=="classic") {
    theta_star = (-1)^seq(1, p) * 2 * exp(-0.7 * seq(1, p))
  } else if(true_param=="inc") {
    # increasing theta star 
    theta_star = seq(-3, 3, length.out=p)
  } else if(true_param=="usc") {
    theta_star = seq(0, 1, length.out=p)
  } else if(true_param=="sparse") {
    theta_star = (-1)^seq(1, p) * 2 * exp(-0.7 * seq(1, p))
    i = tail(1:p, 0.8 * p)
    theta_star[i] <- 0
  } else if(true_param=="single") {
    theta_star <- rep(0, p)
    theta_star[1] = theta_coeff
  } else {
    stop("implement additional theta_star definitions")
  }
  Sigma_X = NA
  if(sigma_x=="id") {
    Sigma_X = diag(p)
  } else if(sigma_x=="equicor") {
    Sigma_X = (1 - rho) * diag(p) + rho * matrix(1, nrow=p, ncol=p) 
  } else if (sigma_x=="ill_cond") {
    lam = seq(0.1, 100, length.out=p)
    # Q = qr.Q(qr(u %*% t(u)))
    Sigma_X = diag(lam)
  } else if (sigma_x=="toeplitz") {
    Sigma_X = toeplitz(0.5^seq(0, p-1))
  } else {
    # TODO: implement different Sigma definitions here.
    stop("implement additional Sigma_X definition")
  } 
  if(nrow(Sigma_X) != ncol(Sigma_X)) {
    stop("Need to feed square matrix here.")
  }
  ##
  if (any(is.na(sigma_sqrt))) {
    S = Sigma_X
    if(sigma_x != "id") {
      S =  sqrtm(Sigma_X)
    }
  } else {
    S = sigma_sqrt
  }
  Z = matrix(rnorm(N * p), ncol=p)
  X = t(S %*% t(Z))
  # rmvnorm(N, mean=rep(0, p), sigma = Sigma_X) #covariates
  pred = X %*% theta_star  # predictor.
  # 
  glm_link = NA # glm link function. 
  Y = NA
  #
  if(model_name == "gaussian") {
    glm_link = function(x) x
    du_glm_link = function(x) 1
    noise = rnorm(N, sd=sigma_noise)
    Y = glm_link(pred) + noise
  } else if(model_name == "poisson") {
    ## TODO: Poisson Y should not be crazy.
    glm_link = function(x) exp(x)
    du_glm_link = function(x) exp(x)
    Y = rpois(N, lambda=glm_link(pred))
  } else if(model_name=="binomial") {
    glm_link = function(x) expit(x)
    du_glm_link = function(x) du_expit(x)
    Y = rbinom(N, size=1, prob=glm_link(pred))
  } else {
    stop("model not implemented.")
  }
  
  return(list(model=model_name, Sigma_X = Sigma_X, X=X, Y=Y, glm_link=glm_link, 
              du_glm_link=du_glm_link, theta_star=theta_star, Sigma_X_sqrt=S,
              cov=sigma_x, true_param=true_param))
}

fisher <- function(data) {
  ## Calculates Fisher information matrix.
  n = nrow(data$X)
  p = ncol(data$X)
  Fi = matrix(0, nrow=p, ncol=p)
  for(i in 1:n) {
    xi = data$X[i, ]
    yi = data$Y[i]
    etai = sum(xi * data$theta_star)
    gradi = matrix((yi - data$glm_link(etai)) * xi, ncol=1)
    Fi = Fi + gradi %*% t(gradi)
  }
  return(Fi / n)
}

# Additions for Baidu Convergence Diagnostic Experiments
cos.sim <- function(A, B) {
	return( sum(A*B)/(sqrt(sum(A^2)*sum(B^2)) + 1e-8) )
}
ip.sim <- function(A, B) {
	return( sum(A*B) )
}
GMM.kernel <- function(A, B) {
	# Data Transformation
	transform_u <- function(u) {
		z = rep(0, length=2*length(u))
		pos.idx = which(u >  0)
		neg.idx = which(u <= 0)
		z[2*pos.idx-1] = u[pos.idx]
	  z[2*neg.idx]   = -1 * u[neg.idx]	
	  return(z)
	}
	A_u = transform_u(A) 
	B_u = transform_u(B)

	GMM = sum(pmin(A_u, B_u)) / sum(pmax(A_u, B_u))
	return(GMM)
}
covar.sim <- function(A, B) {
	return( sum( (A-mean(A)) * (B-mean(B)) ))
}

init_opt <- function(data, theta0, gamma, nepoch, beta, batch_size, momentum.switch) {
	#args
	args = list()
	if ('X' %in% names(data)) {
		N = nrow(data$X)
		p = ncol(data$X)
	} else {
		N = nrow(data$A)
		p = ncol(data$A)
	}
  if (is.na(theta0)) {
    theta0 = rnorm(p)
		#theta0 = rnorm(p, sd=0.1)
		#theta0 = rep(0, p)
  }
	args$theta0  = theta0
	args$gamma = gamma
	args$beta = beta
  args$n_batch = as.integer(N / batch_size) 
	args$batch_size = batch_size
	args$nepoch = nepoch
  args$momentum = 0
  args$momentum.prev = 0
  args$grad.prev = 0
  args$mean.burnin_done = FALSE
  args$mean.burnin = 0
  args$mean.convg = FALSE
	args$momentum.switch=FALSE
	if (momentum.switch==FALSE) { args$momentum.switch = -1 }
	args$mom.norm.thresh = 0.2 #0.2 threshold for percent change of gradient norm as momentum switching point
	args$final.beta = 0.1 # final beta after momentum reduce

	#logging output
	out = list()
	out$theta_all     = matrix(0, nrow=p, ncol=args$n_batch*nepoch+1)
  out$theta_all[, 1]= theta0
  out$mse_all       = vector("numeric", length=args$n_batch*nepoch+1)
  out$mse_all[1]    = mean( (theta0 - data$theta_star)^2 )
	out$mse_mean      = vector("numeric", length=nepoch)
  #test_all     = vector("numeric", length=n_batch*nepoch)
  #test_mean    = vector("numeric", length=nepoch)
  out$cos_loss_all  = vector("numeric", length=args$n_batch*nepoch)
  out$cos_loss_mean = vector("numeric", length=nepoch)
  out$cos_loss_std  = vector("numeric", length=nepoch)
  out$cos_opt_all   = vector("numeric", length=args$n_batch*nepoch)
  out$cos_opt_mean  = vector("numeric", length=nepoch)
  out$cos_opt_std   = vector("numeric", length=nepoch)
  out$ip_loss_all   = vector("numeric", length=args$n_batch*nepoch)
  out$ip_loss_mean  = vector("numeric", length=nepoch)
  out$ip_loss_std   = vector("numeric", length=nepoch)
  out$ip_opt_all    = vector("numeric", length=args$n_batch*nepoch)
  out$ip_opt_mean   = vector("numeric", length=nepoch)
  out$ip_opt_std    = vector("numeric", length=nepoch)
	out$grad_norm_all = vector("numeric", length=args$n_batch*nepoch)
	out$grad_norm_mean= vector("numeric", length=nepoch)
	out$grad_norm_std = vector("numeric", length=nepoch)
  out$gmm_loss_all  = vector("numeric", length=args$n_batch*nepoch)
  out$gmm_loss_mean = vector("numeric", length=nepoch)
  out$gmm_loss_std  = vector("numeric", length=nepoch)
  out$gmm_opt_all   = vector("numeric", length=args$n_batch*nepoch)
  out$gmm_opt_mean  = vector("numeric", length=nepoch)
  out$gmm_opt_std   = vector("numeric", length=nepoch)
	# Trying out
	out$momGrad_all   = vector("numeric", length=args$n_batch*nepoch)
	out$momGrad_mean  = vector("numeric", length=nepoch)
	
	return(list(args=args, out=out))
}

batch_udpate <- function(j, data, gradn, params) {
	args = params$args
	out  = params$out

	args$momentum = args$beta * args$momentum + args$gamma * gradn / args$batch_size
      
	out$theta_all[, j] = out$theta_all[, j-1] - args$momentum #- weight_decay * theta_all[, j-1]
	out$mse_all[j]     = mean( (out$theta_all[, j] - data$theta_star)^2 )

	out$cos_loss_all[j-1]  = cos.sim(gradn, args$grad.prev)
	out$cos_opt_all[j-1]   = cos.sim(args$momentum, args$momentum.prev)
	out$ip_loss_all[j-1]   = ip.sim(gradn, args$grad.prev)
	out$ip_opt_all[j-1]    = ip.sim(args$momentum, args$momentum.prev)
	out$grad_norm_all[j-1] = sum(gradn^2)
	out$momGrad_all[j-1]   = cos.sim(args$gamma * gradn / args$batch_size, args$momentum)
	#out$gmm_loss_all[j-1]  = GMM.kernel(gradn, args$grad.prev)
	#out$gmm_opt_all[j-1]   = GMM.kernel(args$momentum, args$momentum.prev)
	out$gmm_loss_all[j-1]  = covar.sim(gradn, args$grad.prev)
	out$gmm_opt_all[j-1]   = covar.sim(args$momentum, args$momentum.prev)
	#out$momGrad_all[j-1]   = sum((args$momentum - args$gamma * gradn / args$batch_size)^2)
	args$momentum.prev = args$momentum
	args$grad.prev = gradn
	
	return(list(args=args, out=out))
}

epoch_update <- function(epoch, params) {
	args = params$args
	out  = params$out

	start = (epoch-1) * args$n_batch + 1
	end   = epoch * args$n_batch
	out$mse_mean[epoch]      = mean(out$mse_all[start:end])
	out$cos_loss_mean[epoch] = mean(out$cos_loss_all[start:end])
	out$cos_loss_std[epoch]  = sd(out$cos_loss_all[start:end])
	out$cos_opt_mean[epoch]  = mean(out$cos_opt_all[start:end])
	out$cos_opt_std[epoch]   = sd(out$cos_opt_all[start:end])
	out$ip_loss_mean[epoch]  = mean(out$ip_loss_all[start:end])
	out$ip_loss_std[epoch]   = sd(out$ip_loss_all[start:end])
	out$ip_opt_mean[epoch]   = mean(out$ip_opt_all[start:end])
	out$ip_opt_std[epoch]    = sd(out$ip_opt_all[start:end])
	out$grad_norm_mean[epoch]= mean(out$grad_norm_all[start:end])
	out$grad_norm_std[epoch] = sd(out$grad_norm_all[start:end])
	out$gmm_loss_mean[epoch] = mean(out$gmm_loss_all[start:end])
	out$gmm_loss_std[epoch]  = sd(out$gmm_loss_all[start:end])
	out$gmm_opt_mean[epoch]  = mean(out$gmm_opt_all[start:end])
	out$gmm_opt_std[epoch]   = sd(out$gmm_opt_all[start:end])
	# Trying out
	out$momGrad_mean[epoch]  = mean(out$momGrad_all[start:end])

	# Momentum Reduction
	if (epoch > 1 & args$momentum.switch==FALSE) {
		# norm heuristic
		if (abs(out$grad_norm_mean[epoch] - out$grad_norm_mean[epoch-1])/out$grad_norm_mean[epoch] 
				< args$mom.norm.thresh) {
			args$momentum.switch = epoch
			args$mean.burnin = epoch
			args$beta = args$final.beta
			print(sprintf("momentum parameter reduced to:%0.2f at epoch:%d", args$beta, epoch))
		} # Automatic Burnin (if no momentum reduction)
	} else if (args$momentum.switch==-1 & args$mean.burnin_done==FALSE) {
		if (out$ip_loss_mean[epoch] < 0) {
			args$mean.burnin_done = TRUE
			args$mean.burnin = epoch
			print(sprintf("mean burnin done at epoch:%d", epoch))
		}
	}	
	# Convgergence Diagnostic
	if ((args$momentum.switch>1 | args$mean.burnin_done==TRUE) & args$mean.convg==FALSE) {
		if (mean(tail(out$ip_loss_all, (args$nepoch-args$mean.burnin)*args$n_batch)) < 0) {
			args$mean.convg = epoch
			print(sprintf("Convergence diagnostic activated at epoch:%d", epoch))
		}	
	}

	return(list(args=args, out=out))
}

plot_diag <- function(j, params) {
	args = params$args
	out  = params$out
	#par(mfrow=c(4,2))
	par(mfrow=c(3,2))

  mean.final_nepoch = (args$nepoch-args$mean.burnin)
  mean.x = seq(args$mean.burnin, args$nepoch, length.out=mean.final_nepoch*args$n_batch)

	# MSE
  mse.x = seq(2, args$nepoch, length.out=j-1)
  plot(mse.x, out$mse_all, type='l', xlab='epoch', ylab='MSE', main='MSE of SGD w/Momentum estimator')
	if (is.numeric(args$momentum.switch)) {abline(v=args$momentum.switch, col='blue', lty=2)}
  if (args$mean.burnin_done) {abline(v=args$mean.burnin, col='red', lty=3)}
  if (is.numeric(args$mean.convg)) {abline(v=args$mean.convg, col='red', lty=2)}
  #if (args$skew.burnin_done) {abline(v=args$skew.burnin, col='blue', lty=3)}
  #if (is.numeric(args$skew.convg)) {abline(v=args$skew.convg, col='blue', lty=2)}
  legend(x='topright', legend=c('momentum.switch', 'mean.burnin done', 'mean.diagnostic\n activated'),
				 col=c('blue','red','red'), lty=c(2,3,2))
 
	# BoxPlots
	#boxplot(tail(out$ip_loss_all, final_nepoch*args$n_batch), 1, main='Boxplots for IP and COS after burnin')
	#axis(side=1, labels=c('IP', 'COS'), at=c(1,2))
	#par(new=TRUE)
	#boxplot(1, tail(out$cos_loss_all, final_nepoch*args$n_batch),
	#				axes=FALSE, xlab=NA, ylab=NA)
	#axis(side=4)
	#mtext(side=4, line=3, 'COS')

	# Sorted magnitudes IP	
	l = quantile(tail(out$ip_loss_all, mean.final_nepoch*args$n_batch), 0.01)
	u = quantile(tail(out$ip_loss_all, mean.final_nepoch*args$n_batch), 0.99)
  plot(sort(tail(out$ip_loss_all, mean.final_nepoch*args$n_batch)), type='l', ylim=c(l,u), 
			 ylab='IP Loss', xlab='distribution of gradient samples', xaxt='n', main='Sorted IP Loss')
	abline(h=0, col='grey')
  #abline(h=mean(tail(out$ip_loss_mean, final_nepoch)), col='red')
	#legend(x='topleft', legend=c(sprintf('mean after\nburnin:%.3f', mean(tail(out$ip_loss_mean, final_nepoch)))), 
	#                             col='red', lty=1)
	legend(x='topleft', legend=c('After mean.burnin', 'Before mean.burnin'), col=c('black', 'blue'), lty=1)
	if (args$mean.burnin_done | args$momentum.switch>1) {
		par(new=TRUE)
		l = quantile(head(out$ip_loss_all, args$mean.burnin*args$n_batch), 0.01); u = quantile(head(out$ip_loss_all, args$mean.burnin*args$n_batch), 0.99)
		plot(sort(head(out$ip_loss_all, args$mean.burnin*args$n_batch)), type='l', col='blue', ylim=c(l,u),
				 axes=FALSE, xlab=NA, ylab=NA)
		axis(side=4, col='blue')
		abline(h=0, col='lightblue')
	}

	## Sorted magnitudes COS	
	#l = quantile(tail(out$cos_loss_all, final_nepoch*args$n_batch), 0.01); u = quantile(tail(out$cos_loss_all, final_nepoch*args$n_batch), 0.99)
  #plot(sort(tail(out$cos_loss_all, final_nepoch*args$n_batch)), type='l', ylim=c(l,u), 
	#		 ylab='COS Loss', xlab='distribution of gradient samples', xaxt='n', main='Sorted COS Loss')
	##abline(h=0, col='grey')
  ##abline(h=mean(tail(out$cos_loss_mean, final_nepoch)), col='red')
	##legend(x='topleft', legend=c(sprintf('mean after\nburnin:%.3f',mean(tail(out$cos_loss_mean, final_nepoch)))),
	##       col='red', lty=1)
	#legend(x='topleft', legend=c('After burnin', 'Before burnin'), col=c('black', 'blue'), lty=1)
	#if (args$burnin_done) {
	#	par(new=TRUE)
	#	l = quantile(head(out$cos_loss_all, args$burnin*args$n_batch), 0.01); u = quantile(head(out$cos_loss_all, args$burnin*args$n_batch), 0.99)
	#	plot(sort(head(out$cos_loss_all, args$burnin*args$n_batch)), type='l', col='blue', ylim=c(l,u),
	#			 axes=FALSE, xlab=NA, ylab=NA)
	#	axis(side=4, col='blue')
	#	abline(h=0, col='lightblue')
	#}

	# Dissociate Norm and Cos
	plot(head(out$cos_loss_all, args$mean.burnin*args$n_batch), 
			 head(out$grad_norm_all, args$mean.burnin*args$n_batch), 
			 #abs(head(out$ip_loss_all, args$mean.burnin*args$n_batch)),
			 xlab='cosine (angle) successive gradients', ylab='gradient norm', main='Angle vs Norm Pre Burnin')
	abline(v=0, col='red', lty=2)

	plot(tail(out$cos_loss_all, mean.final_nepoch*args$n_batch), 
			 tail(out$grad_norm_all, mean.final_nepoch*args$n_batch),
			 #abs(tail(out$ip_loss_all, mean.final_nepoch*args$n_batch)),
			 xlab='cosine (angle) successive gradients', ylab='gradient norm', main='Angle vs Norm Post Burnin')
	abline(v=0, col='red', lty=2)

	# IP Loss	
	l = quantile(out$ip_loss_mean, 0.01); u = quantile(out$ip_loss_mean, 0.97)
  plot(out$ip_loss_mean, type='b', ylim=c(l,u),
       xlab='epoch', ylab=sprintf('IP mean per epoch'),
       main=sprintf('Per epoch mean of IP gradient loss'))
  abline(h=0)
  abline(h=mean(tail(out$ip_loss_mean, mean.final_nepoch)), col='red')
  legend(x='topright', legend=sprintf('Mean after\nmean.burnin: %.2f', 
																			mean(tail(out$ip_loss_mean, mean.final_nepoch))),
				 col='red', lty=1)
  
	## COS Loss	
	#l = quantile(out$cos_loss_mean, 0.05); u = quantile(out$cos_loss_mean, 0.95)
  #plot(out$cos_loss_mean, type='l', ylim=c(l,u),
  #     xlab='epoch', ylab=sprintf('COS mean per epoch'),
  #     main=sprintf('Per epoch mean of COS gradient loss'))
  #abline(h=0)
  #abline(h=mean(tail(out$cos_loss_mean, final_nepoch)), col='red')
  #legend(x='topright', legend=sprintf('Mean after\nburnin: %.2f', mean(tail(out$cos_loss_mean, final_nepoch))),
	#			 col='red', lty=1)

	# OPT
  #plot(out$ip_opt_mean, type='l', col='blue', xlab='epoch', ylab=sprintf('IP mean per epoch'),
  #     main=sprintf('Per epoch mean of IP gradient loss + momentum'))
  #legend(x="topright", legend=c("loss", "opt"), col=c("red", "blue"), lty=1)
 
	# CUMSUM IP
  plot(mean.x, cumsum(tail(out$ip_loss_all, mean.final_nepoch*args$n_batch)), type='l', 
       xlab='epoch', ylab=sprintf('IP total sum after burnin'),
       main='Convergence diagnostic IP test statistic: activate when < 0')
  abline(h=0)

	# CUMSUM COS	
  #plot(x, cumsum(tail(out$cos_loss_all, final_nepoch*args$n_batch)), type='l', 
  #     xlab='epoch', ylab=sprintf('COS total sum after burnin'),
  #     main='Convergence diagnostic COS test statistic: activate when < 0')
  #abline(h=0)

	# IP Mom and Grad
	#plot(out$momGrad_mean, type='l', log='y',
	#		 xlab='epoch', ylab='IP between gradn & momn', main='Determine when to switch of Mom')

	print(sprintf("MSE after mean.convg: %.3f", out$mse_all[args$mean.convg*args$n_batch]))
  par(mfrow=c(1,1))

}

plot_GMM <- function(j, params) {
	args = params$args
	out  = params$out
  mean.final_nepoch = (args$nepoch-args$mean.burnin)
  mean.x = seq(args$mean.burnin, args$nepoch, length.out=mean.final_nepoch*args$n_batch)

	par(mfrow=c(3,2))

	# MSE
  mse.x = seq(2, args$nepoch, length.out=j-1)
  plot(mse.x, out$mse_all, type='l', xlab='epoch', ylab='MSE', main='MSE of SGD w/Momentum estimator')
	#if (is.numeric(args$momentum.switch)) {abline(v=args$momentum.switch, col='blue', lty=2)}
  #if (args$mean.burnin_done) {abline(v=args$mean.burnin, col='red', lty=3)}
  #if (is.numeric(args$mean.convg)) {abline(v=args$mean.convg, col='red', lty=2)}
  #if (args$skew.burnin_done) {abline(v=args$skew.burnin, col='blue', lty=3)}
  #if (is.numeric(args$skew.convg)) {abline(v=args$skew.convg, col='blue', lty=2)}
  legend(x='topright', legend=c('momentum.switch', 'mean.burnin done', 'mean.diagnostic\n activated'),
				 col=c('blue','red','red'), lty=c(2,3,2))

	# Sorted magnitudes cos	
	l = quantile(tail(out$cos_loss_all, mean.final_nepoch*args$n_batch), 0.01)
	u = quantile(tail(out$cos_loss_all, mean.final_nepoch*args$n_batch), 0.99)
  plot(sort(tail(out$cos_loss_all, mean.final_nepoch*args$n_batch)), type='l', ylim=c(l,u), 
			 ylab='cos Loss', xlab='distribution of gradient samples', xaxt='n', main='Sorted cos Loss')
	abline(h=0, col='grey')
  #abline(h=mean(tail(out$cos_loss_mean, final_nepoch)), col='red')
	#legend(x='topleft', legend=c(sprintf('mean after\nburnin:%.3f', mean(tail(out$cos_loss_mean, final_nepoch)))), 
	#                             col='red', lty=1)
	legend(x='topleft', legend=c('After mean.burnin', 'Before mean.burnin'), col=c('black', 'blue'), lty=1)
	if (args$mean.burnin_done | args$momentum.switch>1) {
		par(new=TRUE)
		l = quantile(head(out$cos_loss_all, args$mean.burnin*args$n_batch), 0.01); u = quantile(head(out$cos_loss_all, args$mean.burnin*args$n_batch), 0.99)
		plot(sort(head(out$cos_loss_all, args$mean.burnin*args$n_batch)), type='l', col='blue', ylim=c(l,u),
				 axes=FALSE, xlab=NA, ylab=NA)
		axis(side=4, col='blue')
		abline(h=0, col='lightblue')
	}

	# Sorted magnitudes gmm	
	l = quantile(tail(out$gmm_loss_all, mean.final_nepoch*args$n_batch), 0.01)
	u = quantile(tail(out$gmm_loss_all, mean.final_nepoch*args$n_batch), 0.99)
  plot(sort(tail(out$gmm_loss_all, mean.final_nepoch*args$n_batch)), type='l', ylim=c(l,u), 
			 ylab='gmm Loss', xlab='distribution of gradient samples', xaxt='n', main='Sorted gmm Loss')
	abline(h=0, col='grey')
  #abline(h=mean(tail(out$gmm_loss_mean, final_nepoch)), col='red')
	#legend(x='topleft', legend=c(sprintf('mean after\nburnin:%.3f', mean(tail(out$gmm_loss_mean, final_nepoch)))), 
	#                             col='red', lty=1)
	legend(x='topleft', legend=c('After mean.burnin', 'Before mean.burnin'), col=c('black', 'blue'), lty=1)
	if (args$mean.burnin_done | args$momentum.switch>1) {
		par(new=TRUE)
		l = quantile(head(out$gmm_loss_all, args$mean.burnin*args$n_batch), 0.01); u = quantile(head(out$gmm_loss_all, args$mean.burnin*args$n_batch), 0.99)
		plot(sort(head(out$gmm_loss_all, args$mean.burnin*args$n_batch)), type='l', col='blue', ylim=c(l,u),
				 axes=FALSE, xlab=NA, ylab=NA)
		axis(side=4, col='blue')
		abline(h=0, col='lightblue')
	}

	# Sorted magnitudes IP	
	l = quantile(tail(out$ip_loss_all, mean.final_nepoch*args$n_batch), 0.01)
	u = quantile(tail(out$ip_loss_all, mean.final_nepoch*args$n_batch), 0.99)
  plot(sort(tail(out$ip_loss_all, mean.final_nepoch*args$n_batch)), type='l', ylim=c(l,u), 
			 ylab='IP Loss', xlab='distribution of gradient samples', xaxt='n', main='Sorted IP Loss')
	abline(h=0, col='grey')
  #abline(h=mean(tail(out$ip_loss_mean, final_nepoch)), col='red')
	#legend(x='topleft', legend=c(sprintf('mean after\nburnin:%.3f', mean(tail(out$ip_loss_mean, final_nepoch)))), 
	#                             col='red', lty=1)
	legend(x='topleft', legend=c('After mean.burnin', 'Before mean.burnin'), col=c('black', 'blue'), lty=1)
	if (args$mean.burnin_done | args$momentum.switch>1) {
		par(new=TRUE)
		l = quantile(head(out$ip_loss_all, args$mean.burnin*args$n_batch), 0.01); u = quantile(head(out$ip_loss_all, args$mean.burnin*args$n_batch), 0.99)
		plot(sort(head(out$ip_loss_all, args$mean.burnin*args$n_batch)), type='l', col='blue', ylim=c(l,u),
				 axes=FALSE, xlab=NA, ylab=NA)
		axis(side=4, col='blue')
		abline(h=0, col='lightblue')
	}

	# GMM Loss	
	l = quantile(out$gmm_loss_mean, 0.01); u = quantile(out$gmm_loss_mean, 0.97)
  plot(out$gmm_loss_mean, type='b', ylim=c(l,u),
       xlab='epoch', ylab=sprintf('gmm mean per epoch'),
       main=sprintf('Per epoch mean of gmm gradient loss'))
  abline(h=0)

	# IP Loss	
	l = quantile(out$ip_loss_mean, 0.01); u = quantile(out$ip_loss_mean, 0.97)
  plot(out$ip_loss_mean, type='b', ylim=c(l,u),
       xlab='epoch', ylab=sprintf('ip mean per epoch'),
       main=sprintf('Per epoch mean of ip gradient loss'))
  abline(h=0)


	par(mfrow=c(1,1))
}

plot_phase <- function(params, epoch, range=1) {
	args = params$args
	out = params$out

	norm <- function(v) {return(v / sqrt(sum(v^2)))}

	start = (epoch-1) * args$n_batch + 1
	end   = (epoch-1+range) * args$n_batch
	
	par(mfrow=c(2,1))
	plot(sort(out$ip_loss_all[start:end]), type='l') 
	hist(out$ip_loss_all[start:end])
	print(skewness(out$ip_loss_all[start:end]))
}

compute_skew <- function(params, epoch, range=1) {
	args = params$args
	out = params$out

	start = (epoch-1) * args$n_batch + 1
	end   = (epoch-1+range) * args$n_batch
	
	print(skewness(out$ip_loss_all[start:end]))
}

plot_phase_2 <- function(params) {
	args = params$args
	out = params$out

	norm <- function(v) {return(v / sqrt(sum(v^2)))}

	start = 1
	end = args$n_batch
	plot(norm(sort(out$ip_loss_all[start:end])), type='l', ylim=c(-1,1), col=grey(1-1/20)) 

	for (epoch in 1:20) {
		start = (epoch-1) * args$n_batch + 1
		end   = epoch * args$n_batch
		
		lines(norm(sort(out$ip_loss_all[start:end])), col=grey(1-epoch/20))

		Sys.sleep(1)
	}
}

plot_ip <- function(params, breakpoint_epoch, plot_bool=FALSE) {
  if (plot_bool) {
    pdf("HistIP.pdf", width=8, height=11)
  }
  
  args = params$args
  out  = params$out
  par(mfrow=c(2,1))
  par(mar=c(5.1,5.1,4.1,2.1))
  
  breakpoint_iter = round(breakpoint_epoch * args$n_batch)
  # Transient
  ip.trans = out$ip_loss_all[1:breakpoint_iter]
  #plot(sort(ip.trans))
  hist(ip.trans, breaks=30, col='lightgray', 
       cex.main=2, cex.axis=2, cex.lab=2, ,
       xlab="Inner Product", main="Histogram of IP's in Transient Phase")
  
  # Stationary
  ip.stat = out$ip_loss_all[breakpoint_iter:length(out$ip_loss_all)]
  #plot(sort(ip.stat))
  hist(ip.stat, breaks=30, col='lightgray', 
       cex.main=2, cex.axis=2, cex.lab=2,
       xlab="Inner Product", main="Histogram of IP's in Stationary Phase")
  abline(v=mean(ip.stat), col='red', lty=2, lwd=5)
  legend(x='topleft', legend=sprintf("Mean IP\nin Stationarity:\n%.2f", mean(ip.stat)), col='red', lty=2, lwd=5, cex=1.8,
         bty='n', bg='white')
  
  par(mfrow=c(1,1))
  par(mar=c(5.1,4.1,4.1,2.1))
  if (plot_bool) {
    dev.off()
  }
}

plot_angle_norm <- function(params, circle.xy, circle.rad=c(0.2,200), plot_bool=FALSE) {
  if (plot_bool) {
    pdf("4-2_AngleNorm.pdf", width=8, height=8)
  }
  
  args = params$args
  out  = params$out
  par(mar=c(5.1,5.1,4.1,2.1))
  
  mean.final_nepoch = (args$nepoch-args$mean.burnin)
  plot(tail(out$cos_loss_all, mean.final_nepoch*args$n_batch), 
       tail(out$grad_norm_all, mean.final_nepoch*args$n_batch),
       pch=16, cex.main=2, cex.axis=2, cex.lab=2,
       xlab='Cosine (angle) of successive gradients', ylab='Gradient norm', main='Angle vs Norm in Stationary Phase')
  abline(v=0, col='black', lty=1)
  draw.circle(circle.xy[1], circle.xy[2], border='red', radius=circle.rad, lwd=3.5)
  legend(x='topright', legend='Key iterates with\nnegative angle and\nhigh gradient norm', 
         col='red', lty=1, lwd=3.5, cex=1.35, bty='n')
  
  if (plot_bool) {
    dev.off()
  }
}


plot_mse_ip <- function(params, epoch_ls) {
	args = params$args
	out  = params$out
	par(mfrow=c(1,3))

	# MSE
  mse.x = seq(2, args$nepoch, length.out=length(out$mse_all))
  plot(mse.x, out$mse_all, type='l', xlab='epoch', ylab='MSE', main='MSE of SGD with Momentum')
	abline(v=epoch_ls, col='blue', lty=3)
	y_txt = out$mse_all[2] * 0.7
	x_txt = epoch_ls[1]/2 + 0.7 
	textbox(x=c(x_txt-0.6,x_txt+0.6), y=y_txt, textlist='1', font=20, fill='white')
	x_txt = epoch_ls[1] + (epoch_ls[2]-epoch_ls[1])/2
	textbox(x=c(x_txt-0.6,x_txt+0.6), y=y_txt, textlist='2', fill='white')
	#x_txt = epoch_ls[2] + (epoch_ls[3]-epoch_ls[2])/2
	#textbox(x=c(x_txt-0.6,x_txt+0.6), y=y_txt, textlist='3')

	# Transient phase sorted IP
	start = 1; end = epoch_ls[1] * args$n_batch

	# Sorted magnitudes IP	
  plot(sort(out$ip_loss_all[start:end]), type='l',  
			 ylab='IP of successive gradients', xlab='', 
			 xaxt='n', main='Region 1 sorted IP')
	abline(h=0, col='black')
  abline(h=mean(out$ip_loss_all[start:end]), col='red', lty=2)
	legend(x='topleft', legend=c('IP mean'), col='red', lty=2)

	# Phase transition sorted IP
	start = epoch_ls[1]*args$n_batch+1; end = epoch_ls[2] * args$n_batch
  plot(sort(out$ip_loss_all[start:end]), type='l',  
			 ylab='IP of successive gradients', xlab='', xaxt='n', main='Region 2 sorted IP')
	abline(h=0, col='black')
  abline(h=mean(out$ip_loss_all[start:end]), col='red', lty=2)
	legend(x='topleft', legend=c('IP mean'), col='red', lty=2)

	# Stationary phase sorted IP
	#start = epoch_ls[2]*args$n_batch+1; end = epoch_ls[3] * args$n_batch
  #plot(sort(out$ip_loss_all[start:end]), type='l',  
	#		 ylab='IP of successive gradients', xlab='', xaxt='n', main='Region 3 sorted IP')
	#abline(h=0, col='black')
  #abline(h=mean(out$ip_loss_all[start:end]), col='red', lty=2)
	#legend(x='topleft', legend=c('IP mean'), col='red', lty=2)

	par(mfrow=c(1,1))
}



convg_eval <- function(params, convg=NA, eta=5e-3) {
	args = params$args
	out  = params$out

	if(!is.na(convg)) {
		args$mean.convg = convg
	}

	# If too early
	convg.mse = out$mse_mean[args$mean.convg] 
	best.mse  = min(out$mse_mean[args$mean.convg:args$nepoch]) 
	delta.mse = convg.mse - best.mse
	if (delta.mse < 0) {
		print("delta.mse is negative")
	}

	# If too late
	k = args$mean.convg - which.min(abs(out$mse_mean[1:args$mean.convg] - convg.mse - eta))
	kT = k / args$mean.convg

	return(list(convg.mse=convg.mse, best.mse=best.mse, delta.mse=delta.mse, k=k, kT=kT))
}


mult_convg_analysis <- function(mult, nepoch, beta, gamma, batch_size, gen_dat, sgdm,
																convg=NA, eta=5e-3, kappa=0.8, nonconvex=TRUE) {
	delta.mse.ls = c()
	kT.ls        = c()
	convg.ls     = c()

	for (i in 1:mult) {
		keep.run = TRUE
		while (keep.run & nonconvex) {
			dat    = gen_dat()
			params = sgdm(dat, nepoch=nepoch, beta=beta, gamma=gamma, batch_size=batch_size, verbose=FALSE)
			# prevents bad minima for phase retrieval case 
			keep.run = ! (tail(params$out$mse_mean,1) < 0.05)	
		}

		eval = convg_eval(params, convg, eta)
	
		delta.mse.ls = c(delta.mse.ls, eval$delta.mse)
		kT.ls        = c(kT.ls, eval$kT)
		convg.ls     = c(convg.ls, params$args$mean.convg)
	}

	num.too.early = sum(delta.mse.ls > eta)
	num.too.late  = sum(kT.ls > kappa)
	num.good.diag = mult - num.too.early - num.too.late

	return(list(delta.mse=delta.mse.ls, kT=kT.ls, convg=convg.ls,
							too.early=num.too.early, too.late=num.too.late, good.diag=num.good.diag))
}
