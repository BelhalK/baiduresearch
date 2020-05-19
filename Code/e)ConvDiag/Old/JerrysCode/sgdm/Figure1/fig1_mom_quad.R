rm(list=ls())
source("fig1_lib.R")


sgd_momentum <- function(data, theta0=NA, gamma=0.0005, nepoch=1, beta=0.9, batch_size=1, momentum.switch=TRUE, verbose=TRUE) {
  params = init_opt(data=data, theta0=theta0, gamma=gamma, nepoch=nepoch, beta=beta, batch_size=batch_size, momentum.switch=momentum.switch)
  N = nrow(data$X)
  j = 2
  for(epoch in 1:nepoch) {
    idx = sample(1:N)
    i = 1
    for(n in 1:(params$args$n_batch)) {
      gradn = 0
      # mini-batch loop
      for (k in 1:batch_size) {
        xi = data$X[idx[i], ]
        yi = data$Y[idx[i]]
        predi = sum(params$out$theta_all[, j-1] * xi)
        gradn = gradn - (yi - data$glm_link(predi)) * xi
        i = i + 1
      }
      params = batch_udpate(j, data, gradn, params)
      j = j + 1
    }
    params = epoch_update(epoch, params)
  }
  if (verbose) {
    plot_diag(j, params)
  }
  return(params)
}

## to run new examples
dat = gen_data(sigma_x='id', sigma_noise=1)
params = sgd_momentum(dat, nepoch=20, beta=0.8, gamma=1e-2, batch_size=20, momentum.switch=FALSE)
#args = params$args
#out  = params$out

## to same parameters
#saveRDS(params, "params_fname.rds")

# load parameters
params = readRDS("params_07-15-19.rds")

# for Figure 1
# second argument is epoch for convergence. Can double check lookin in params$out
plot_ip(params, 7, plot_bool=TRUE)







