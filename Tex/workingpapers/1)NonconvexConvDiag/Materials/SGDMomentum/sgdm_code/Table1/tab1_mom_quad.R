#rm(list=ls())
source("tab1_lib.R")

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
    #plot_GMM(j, params)
  }
  return(params)
}

mult_runs <- function(m=100, gamma, nepoch, beta, batch_size) {
  m.ip = c()
  for (i in 1:m) {
    dat    = gen_data(sigma_x='id')
    params = sgd_momentum(dat, gamma=gamma, nepoch=nepoch, beta=beta, batch_size=batch_size, momentum.switch=FALSE, verbose=FALSE)
    args   = params$args
    out    = params$out
    m.ip   = c(m.ip, mean(tail(out$ip_loss_all, (args$nepoch-args$mean.burnin)*args$n_batch)))
  }
  return(list(m.ip=mean(m.ip), m.ip.ls=m.ip))
}

# for Table 1, run under 3 settings listed in paper
m = mult_runs(m=25, gamma=1e-2, nepoch=50, beta=0.2, batch_size=20)
m = mult_runs(m=25, gamma=1e-2, nepoch=50, beta=0.8, batch_size=20)

# gives output for mean ip after burnin
m$m.ip
