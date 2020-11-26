library(e1071)
library(cumstats)

expit <- function(x) sapply(x, function(i) { if(i > 60) return(1); if(i < -60) return(0); return(exp(i)/(1 + exp(i)))})
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
  pred = X %*% theta_star  # predictor.
  glm_link = NA # glm link function. 
  Y = NA
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

init_opt <- function(data, theta0, gamma, nepoch, beta, batch_size, 
										 burnin_frac, momentum_switch, autoLR) {
	if ('X' %in% names(data)) {
		N = nrow(data$X)
		p = ncol(data$X)
	} else { # phase retrieval 
		N = nrow(data$A)
		p = ncol(data$A)
	}
  if (is.na(theta0)) {
    theta0 = rnorm(p)
		#theta0 = rnorm(p, sd=0.1)
		#theta0 = rep(0, p)
  }

	# Input Arguments
	args = list()
	args$theta0 = theta0
	args$gamma = gamma
	args$beta = beta
  args$niter_batch = as.integer(N / batch_size) 
	args$batch_size = batch_size
	args$nepoch = nepoch
	args$min_burnin = args$niter_batch * burnin_frac
	args$mom_switch_bool = momentum_switch
	args$mom_switch_threshold = 1e-3
	args$final_beta = 0.2
	args$autoLR = autoLR

	# Intermediate Program Variables
	var = list()
	var$lr = args$gamma
	var$beta = args$beta
	var$gard.prev = 0
	var$burnin = c(0)
	var$convg = c(0)
	var$burnin_done = FALSE 
	var$convg_done = FALSE
	var$momentum = 0
	var$mom_switch = 0
	var$mom_switch_done = FALSE

	# Logging Output
	out = list()
	out$theta_all     = matrix(0, nrow=p, ncol=args$niter_batch*nepoch+1)
  out$theta_all[, 1]= theta0
  out$mse_all       = vector("numeric", length=args$niter_batch*nepoch+1)
  out$mse_all[1]    = mean( (theta0 - data$theta_star)^2 )
  #test_all     = vector("numeric", length=niter_batch*nepoch)
  out$ip_all   = vector("numeric", length=args$niter_batch*nepoch)
	out$grad_norm_all = vector("numeric", length=args$niter_batch*nepoch)
	out$mse_step    = vector("numeric", length=args$niter_batch*nepoch)

	return(list(args=args, var=var, out=out))
}

batch_update <- function(j, epoch, data, gradn, params) {
	args = params$args
	var  = params$var
	out  = params$out
	
	var$momentum = var$beta * var$momentum + var$lr * gradn / args$batch_size
	out$theta_all[, j] = out$theta_all[, j-1] - var$momentum #args$gamma * gradn / args$batch_size
	out$mse_all[j]     = mean( (out$theta_all[, j] - data$theta_star)^2 )

	out$ip_all[j-1]   = ip.sim(gradn, var$grad.prev)
	out$grad_norm_all[j-1] = sum(gradn^2)

	out$mse_step[j-1] = mean( (out$theta_all[, j] - out$theta_all[, j-1])^2 ) 
	var$grad.prev = gradn
	start = max(1, j-21)

	# Momentum Switch
	if (j > args$min_burnin & args$mom_switch_bool==TRUE & var$mom_switch_done==FALSE) {
		if (out$mse_step[j-1] < args$mom_switch_threshold) {
			var$beta = args$final_beta
			var$mom_switch = j-1	
			var$mom_switch_done = TRUE
			print(sprintf("Momentum reduced from %.1f to %.1f at iterate:%d epoch %.2f",
										args$beta, args$final_beta, j-1, (j-1)/args$niter_batch))
		}
	# Burnin
	} else if (j > args$min_burnin & var$burnin_done==FALSE & mean(out$ip_all[start:j-1]) < 0) {
	  var$burnin = c(var$burnin, j-1) 
		var$burnin_done = TRUE
		print(sprintf("Burnin done at iterate:%d, epoch:%.2f", j-1, (j-1)/args$niter_batch))
	# Convgergence Diagnostic
	} else if (var$burnin_done==TRUE & var$convg_done==FALSE) { 
		if (mean(tail(out$ip_all, (length(out$ip_all) - tail(var$burnin,1)))) < 0) {
			var$convg = c(var$convg, j-1)
			var$convg_done = TRUE
			print(sprintf("Convergence diagnostic activated at iterate:%d, epoch:%.2f", j-1, (j-1)/args$niter_batch))
		}
	# Repeat LR	
	} else if(args$autoLR==TRUE & var$convg_done==TRUE & var$lr > 1e-5) {
		var$lr = var$lr * 0.1
		var$burnin_done = FALSE
		var$convg_done  = FALSE
		print(sprintf("Learning Rate reduce x0.1 to:%f", var$lr))
	}	
	
	return(list(args=args, var=var, out=out))
}

plot_diag <- function(j, params) {
	args = params$args
	var  = params$var
	out  = params$out
	
	var$burnin = var$burnin[-1]
	var$convg  = var$convg[-1]

	par(mfrow=c(2,2))

	start = max(var$burnin[1], 1, na.rm=TRUE)
	args$final_niter = length(out$ip_all) - start + 1

	# MSE
  x = seq(0, args$nepoch, length.out=j-1)
  plot(x, out$mse_all, type='l', xlab='epoch', ylab='MSE', main='MSE of SGDM')
	if (var$mom_switch_done==TRUE) {abline(v=var$mom_switch/args$niter_batch, col='forestgreen', lty=4)}
  if (is.numeric(var$burnin)) {abline(v=var$burnin/args$niter_batch, col='red', lty=3)}
  if (is.numeric(var$convg)) {abline(v=var$convg/args$niter_batch, col='red', lty=2)}
  legend(x='topright', legend=c('mom switch', 'burnin done', 'mean.diagnostic\n activated'),
				 col=c('forestgreen','red','red'), lty=c(4,3,2))
 
	# CUMSUM IP
	x2 = seq(start/args$niter_batch, length(out$ip_all)/args$niter_batch, length.out=args$final_niter)
  plot(x2, cumsum(tail(out$ip_all, args$final_niter)), type='l', 
       xlab='epoch', ylab=sprintf('IP total sum after burnin'),
       main='IP test stat: activate when\n cumsum < 0')
  abline(h=0)

	plot(x[-1], out$ip_all, type='l',
			 xlab='epoch', ylab='IP', main='IP for each iterate')
	abline(h=0, lty=3)

	plot(x[-1], log10(out$mse_step), type='l', 
			 xlab='epoch', ylab='log10 mse', main='mse between iterates')


	print(sprintf("MSE at convg: %.3f", out$mse_all[var$convg]))
  par(mfrow=c(1,1))

	return(list(args=args, var=var, out=out))
}








# Scratch
# ====================================================================================
# args$diag_window = round(args$niter_batch / 0.01)
# for diagnostic
# out$Mt = vector("numeric", length=args$niter_batch*nepoch)
# out$var_window  = vector("numeric", length=args$niter_batch*nepoch)
# out$musq_window = vector("numeric", length=args$niter_batch*nepoch)
	
# start = max(j-args$diag_window, 0)
# end   = j-1
# mse_slice = out$mse_step[start:end]
# out$Mt[j-1] = quantile(mse_slice, 0.95) / quantile(mse_slice, 0.05)
# ip_slice = out$ip_all[start:end]
# out$var_window[j-1]  = var(ip_slice)
# out$musq_window[j-1] = max(mean(ip_slice)^2, 1e-2)	

# Var / Mean^2
#y = out$var_window/out$musq_window
#l = quantile(y, 0.05, na.rm=TRUE); u = quantile(y, 0.90, na.rm=TRUE) 
#plot(y, type='l', ylim=c(l,u),
#		 xlab='epoch', ylab='',
#		 main='New IP test stat')
#abline(h=0)
# IP Loss	
#l = quantile(out$ip_mean, 0.01); u = quantile(out$ip_mean, 0.97)
#plot(out$ip_mean, type='b', ylim=c(l,u),
#     xlab='epoch', ylab=sprintf('IP mean per epoch'),
#     main=sprintf('Per epoch mean of IP gradient loss'))
#abline(h=0)
#abline(h=mean(tail(out$ip_mean, final_nepoch)), col='red')
#legend(x='topright', legend=sprintf('Mean after\nburnin: %.2f', mean(tail(out$ip_mean, final_nepoch))),
#			 col='red', lty=1)
batch_udpate_old <- function(j, data, gradn, params) {
	args = params$args
	out  = params$out
    
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

	start = (epoch-1) * args$niter_batch + 1
	end   = epoch * args$niter_batch
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
		if (mean(tail(out$ip_all, (args$nepoch-args$burnin)*args$niter_batch)) < 0) {
			args$convg = epoch
			print(sprintf("Convergence diagnostic activated at epoch:%d", epoch))
		}	
	}

	return(list(args=args, out=out))
}

