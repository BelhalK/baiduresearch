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


# Additions for Baidu Convergence Diagnostic Experiments
ip.sim <- function(A, B) {
	return( sum(A*B) )
}

init_opt <- function(data, theta0, gamma, nepoch, batch_size, burnin_frac) {
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
  args$n_batch = as.integer(N / batch_size) 
	args$batch_size = batch_size
	args$nepoch = nepoch
  args$grad.prev = 0
	args$min_burnin = args$n_batch * burnin_frac
  args$burnin_done = FALSE
  args$burnin = 0
  args$convg = FALSE

	#logging output
	out = list()
	out$theta_all     = matrix(0, nrow=p, ncol=args$n_batch*nepoch+1)
  out$theta_all[, 1]= theta0
  out$mse_all       = vector("numeric", length=args$n_batch*nepoch+1)
  out$mse_all[1]    = mean( (theta0 - data$theta_star)^2 )
	out$mse_mean      = vector("numeric", length=nepoch)
  #test_all     = vector("numeric", length=n_batch*nepoch)
  #test_mean    = vector("numeric", length=nepoch)
  out$ip_all   = vector("numeric", length=args$n_batch*nepoch)
  out$ip_mean  = vector("numeric", length=nepoch)
  out$ip_std   = vector("numeric", length=nepoch)
	out$grad_norm_all = vector("numeric", length=args$n_batch*nepoch)
	out$grad_norm_mean= vector("numeric", length=nepoch)
	out$grad_norm_std = vector("numeric", length=nepoch)
	
	return(list(args=args, out=out))
}

batch_update <- function(j, epoch, data, gradn, params) {
	args = params$args
	out  = params$out
    
	gradn = sign(gradn)	
	out$theta_all[, j] = out$theta_all[, j-1] - args$gamma * gradn / args$batch_size
	out$mse_all[j]     = mean( (out$theta_all[, j] - data$theta_star)^2 )

	out$ip_all[j-1]   = ip.sim(gradn, args$grad.prev)
	out$grad_norm_all[j-1] = sum(gradn^2)
	args$grad.prev = gradn

	# Burnin
	if (j > args$min_burnin & args$burnin_done==FALSE & out$ip_all[j-1] < 0) {
		args$burnin_done = TRUE
		args$burnin = j-1 
		print(sprintf("Burnin done at iterate:%d, epoch:%.2f", j-1, (j-1)/args$n_batch))
	}else if (args$burnin_done==TRUE & args$convg==FALSE) { # Convgergence Diagnostic
		if (mean(tail(out$ip_all, (length(out$ip_all)-args$burnin))) < 0) {
			args$convg = j-1
			print(sprintf("Convergence diagnostic activated at iterate:%d, epoch:%.2f", j-1, (j-1)/args$n_batch))
		}	
	}
	
	return(list(args=args, out=out))
}

plot_diag <- function(j, params) {
	args = params$args
	out  = params$out
	par(mfrow=c(2,1))

  final_nepoch = (args$nepoch-args$burnin)
	final_niter = length(out$ip_all) - args$burnin
	# MSE
  x = seq(0, args$nepoch, length.out=j-1)
  plot(x, out$mse_all, type='l', xlab='epoch', ylab='MSE', main='MSE of SignSGD')
  if (args$burnin_done) {abline(v=args$burnin/args$n_batch, col='red', lty=3)}
  if (is.numeric(args$convg)) {abline(v=args$convg/args$n_batch, col='red', lty=2)}
  legend(x='topright', legend=c('burnin done', 'mean.diagnostic\n activated'),
				 col=c('red','red'), lty=c(3,2))
 
	# IP Loss	
	#l = quantile(out$ip_mean, 0.01); u = quantile(out$ip_mean, 0.97)
  #plot(out$ip_mean, type='b', ylim=c(l,u),
  #     xlab='epoch', ylab=sprintf('IP mean per epoch'),
  #     main=sprintf('Per epoch mean of IP gradient loss'))
  #abline(h=0)
  #abline(h=mean(tail(out$ip_mean, final_nepoch)), col='red')
  #legend(x='topright', legend=sprintf('Mean after\nburnin: %.2f', mean(tail(out$ip_mean, final_nepoch))),
	#			 col='red', lty=1)
  
	# CUMSUM IP
	x2 = seq(args$burnin/args$n_batch, length(out$ip_all)/args$n_batch, length.out=final_niter)
  plot(x2, cumsum(tail(out$ip_all, final_niter)), type='l', 
       xlab='epoch', ylab=sprintf('IP total sum after burnin'),
       main='IP test statistic: activate convergence diagnostic when < 0')
  abline(h=0)

	print(sprintf("MSE at convg: %.3f", out$mse_all[args$convg]))
  par(mfrow=c(1,1))

}

batch_udpate_old <- function(j, data, gradn, params) {
	args = params$args
	out  = params$out
    
	gradn = sign(gradn)	
	out$theta_all[, j] = out$theta_all[, j-1] - args$gamma * gradn / args$batch_size
	out$mse_all[j]     = mean( (out$theta_all[, j] - data$theta_star)^2 )

	out$ip_all[j-1]   = ip.sim(gradn, args$grad.prev)
	out$grad_norm_all[j-1] = sum(gradn^2)
	args$grad.prev = gradn
	
	return(list(args=args, out=out))
}

epoch_update_old <- function(epoch, params) {
	args = params$args
	out  = params$out

	start = (epoch-1) * args$n_batch + 1
	end   = epoch * args$n_batch
	out$mse_mean[epoch]      = mean(out$mse_all[start:end])
	out$ip_mean[epoch]  = mean(out$ip_all[start:end])
	out$ip_std[epoch]   = sd(out$ip_all[start:end])
	out$grad_norm_mean[epoch]= mean(out$grad_norm_all[start:end])
	out$grad_norm_std[epoch] = sd(out$grad_norm_all[start:end])

	# Burnin
	if (args$burnin_done==FALSE & out$ip_mean[epoch]<0) {
		args$burnin_done = TRUE
		args$burnin = epoch
		print(sprintf("Burnin done at epoch:%d", epoch))
	}
	# Convgergence Diagnostic
	if (args$burnin_done==TRUE & args$convg==FALSE) {
		if (mean(tail(out$ip_all, (args$nepoch-args$burnin)*args$n_batch)) < 0) {
			args$convg = epoch
			print(sprintf("Convergence diagnostic activated at epoch:%d", epoch))
		}	
	}

	return(list(args=args, out=out))
}

