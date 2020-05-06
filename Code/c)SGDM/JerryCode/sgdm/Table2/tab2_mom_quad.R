rm(list=ls())
source("tab2_lib.R")

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
params = sgd_momentum(dat, nepoch=20, beta=0.2, gamma=1e-2, batch_size=20, momentum.switch=FALSE)
#args = params$args
#out  = params$out



# run under different settigns for Table 2
mca = mult_convg_analysis(mult=100, nepoch=20, beta=0.8, gamma=1e-2, batch_size=20, gen_data, sgd_momentum, 
                          eta=1e-3, kappa=0.65, nonconvex=FALSE)
