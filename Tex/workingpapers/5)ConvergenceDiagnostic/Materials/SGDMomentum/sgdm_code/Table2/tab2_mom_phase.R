rm(list=ls())
source('tab2_lib.R')
gen_phRe_data <- function(N=1000, p=20, 
                          rho=.15, theta_coeff=1,
                          sigma_noise=1, true_param="classic") {
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
  
  A = matrix(rnorm(N * p), ncol=p)
  Y = (A %*% theta_star)^2  # predictor.
  
  return(list(A=A, Y=Y, theta_star=theta_star))
}
foo_var = FALSE
sgd_momentum <- function(data, theta0=NA, gamma=0.0005, nepoch=1, beta=0.9, batch_size=1, 
                         momentum.switch=TRUE, verbose=TRUE) {
  sgd_momentum_inner <- function(data, theta0, gamma, nepoch, beta, batch_size, momentum.switch, verbose) {
    params = init_opt(data=data, theta0=theta0, gamma=gamma, nepoch=nepoch, beta=beta, batch_size=batch_size, momentum.switch=momentum.switch)
    N = nrow(data$A)
    j = 2
    for(epoch in 1:nepoch) {
      idx = sample(1:N)
      i = 1
      for(n in 1:params$args$n_batch) {
        gradn = 0
        # mini-batch loop
        for (k in 1:batch_size) {
          ai = data$A[idx[i], ]
          yi = data$Y[idx[i]]
          gradn = gradn + (sum((ai * params$out$theta_all[, j-1])^2) - yi) *
            sum(ai * params$out$theta_all[, j-1]) * ai
          i = i + 1
          # NA issue with high momentum
          if (sum(is.na(gradn) & foo_var==FALSE) > 0) {
            return(-1)
          }
        }
        params = batch_udpate(j, data, gradn, params)
        j = j + 1
      }
      params = epoch_update(epoch, params)
    }
    if (verbose) {plot_diag(j, params)}
    return(params)
  }
  params = -1
  while (is.numeric(params)) {
    params = sgd_momentum_inner(data=data, theta0=theta0, gamma=gamma, nepoch=nepoch, beta=beta, batch_size=batch_size,
                                momentum.switch=momentum.switch, verbose=verbose)  
    if (is.numeric(params)) {print("Restart")}
  }
  return(params)
}

## to run new examples
# dat = gen_phRe_data()
# params = sgd_momentum(dat, nepoch=20, beta=0.8, gamma=1e-2, batch_size=20, momentum.switch=TRUE)
# args= params$args
# out = params$out

# run under different settigns for Table 2
mca = mult_convg_analysis(mult=100, nepoch=20, beta=0.8, gamma=1e-2, batch_size=20, gen_phRe_data, sgd_momentum, 
                          eta=1e-2, kappa=0.65, nonconvex=TRUE)
