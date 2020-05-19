setwd("/Users/jerry/Documents/SignSGD/")
rm(list=ls())
source("lib.R")

sgd_sign <- function(data, theta0=NA, gamma=0.0005, nepoch=1, batch_size=1, burnin=1, verbose=TRUE) {
  params = init_opt(data=data, theta0=theta0, gamma=gamma, nepoch=nepoch, batch_size=batch_size, burnin=burnin)
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
      params = batch_update(j, epoch, data, gradn, params)
      j = j + 1
    }
    #params = epoch_update(epoch, params)
  }
  if (verbose) {
    plot_diag(j, params)
  }
  print(j)
  return(params)
}

dat = gen_data(sigma_x='id', sigma_noise=1, model='binomial')
params = sgd_sign(dat, nepoch=10, gamma=0.5, batch_size=10, burnin=2)
args = params$args
out  = params$out