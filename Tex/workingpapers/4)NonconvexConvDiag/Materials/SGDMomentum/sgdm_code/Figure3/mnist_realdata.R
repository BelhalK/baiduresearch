rm(list=ls())
source("fig3_lib.R")
source("rebuttal_plotting.R")
library(Rcpp)
sourceCpp("fig3_sgd.cpp")


momentum_sgd <- function(data, theta0=NA, gamma=0.0005, nepoch=1, beta=0, batch_size=1, 
                 burnin=1, constLR=TRUE, momentum_switch=FALSE, autoLR=FALSE, verbose=TRUE) {
  # wrapper function
  params = momentum_sgd_Cpp(trainX=data$X_train, testX=data$X_test, trainY=data$Y_train, testY=data$Y_test, 
                            model_name=data$model,
                            gamma=gamma, nepoch=nepoch, beta=beta, batch_size=batch_size,
                            burnin=burnin, constLR=constLR, momentum_switch=momentum_switch, 
                            autoLR=autoLR, seed=runif(1)*1e6)
  
  params$out$train_loss = as.vector(params$out$train_loss)
  params$out$test_loss = as.vector(params$out$test_loss)
  params$out$train_acc = as.vector(params$out$train_acc)
  params$out$test_acc = as.vector(params$out$test_acc)
  params$out$ip_all = as.vector(params$out$ip_all)
  params$out$grad_norm_all = as.vector(params$out$grad_norm_all)
  params$out$mse_step = as.vector(params$out$mse_step)
  
  plot_temp(params)
  return(params)
}

# test function
test_mom <- function() {
  dat = gen_data(sigma_x='id', sigma_noise=1, model="binomial")
  dat$X_train = dat$X
  dat$X_test  = dat$X
  dat$Y_train  = dat$Y
  dat$Y_test   = dat$Y
  params = momentum_sgd(dat, nepoch=20, beta=0.5, gamma=0.5, batch_size=10)
  # par(mfrow=c(2,1))
  # plot(params$out$train_loss, type='l', ylab='train loss')
  # plot(params$out$train_acc, type='l', ylab='train acc')
  # par(mfrow=c(1,1))
}

# testing plot function
plot_temp <- function(params) {
  args = params$args
  var  = params$var
  out  = params$out
  
  par(mfrow=c(4,2))
  plot(out$train_loss, type='l', ylab='train loss', main='Train loss')
  plot(out$test_loss, type='l', ylab='test loss', main='Test loss')
  plot(out$train_acc, type='l', ylab='train acc', main='Train acc')
  plot(out$test_acc, type='l', ylab='test acc', main='Test acc')
  
  plot(out$ip_all, type='l',
       main='IP')
  plot(cumsum(tail(out$ip_all, args$niter_batch*args$nepoch - var$burnin[1])), type='l',
       main='CUMSUM IP')
  abline(h=0)
  
  hist(log10(out$mse_step), breaks=30,
       main='mse step')
  hist(log10(out$grad_norm_all), breaks=30,
       main='grad norm')
  
  par(mfrow=c(1,1))
}

# final plotting function
plot_pdf <- function(params1, params2, params3,
                     params1.const, params2.const, params3.const,
                     params1.decr, params2.decr, params3.decr,
                     pdf_bool=FALSE) {
  if (pdf_bool) {
    pdf("mnist_binary.pdf", width=10, height=8)
  }
  
  par(mar=c(5.1,5.1,4.1,2.1))
  
  ll = 5
  x = seq(0, 2, length.out=length(params1$out$test_acc))
  plot(x, params1$out$test_acc, col='royalblue2', type='l', ylim=c(0.85, 0.97), lwd=ll,
       cex.main=2, cex.axis=2, cex.lab=2,
       xlab='Epoch', ylab='Test Accuracy', main='MNIST Binary Logistic Regression')
  grid(col='darkgray', lwd=2, lty=1)
  lines(x, params1$out$test_acc, col='royalblue2', lwd=ll)
  lines(x, params2$out$test_acc, col='forestgreen', lwd=ll)
  lines(x, params3$out$test_acc, col='firebrick3', lwd=ll)
  
  lines(x, params1.decr$out$test_acc, col='purple', lty=2, lwd=ll)
  lines(x, params2.decr$out$test_acc, col='black', lty=2, lwd=ll)
  lines(x, params3.decr$out$test_acc, col='orange', lty=2, lwd=ll)
  
  
  L = legend(x='bottomright', title='Initial LR', 
             legend=c('1.0', '0.1', '0.01', '1.0', '0.1', '0.01'), 
             col=c('royalblue2','forestgreen','firebrick3','purple','black','orange'),
             lty=c(1,1,1,2,2,2), lwd=ll, bty='n', cex=1.5)
  rect(xleft=L$rect$left-L$rect$w-0.02, xright=L$rect$left+L$rect$w, 
       ybottom=L$rect$top-L$rect$h, ytop=L$rect$top, col='white', lwd=1)
  legend(x='bottomright', title='Initial LR', 
             legend=c('1.0', '0.1', '0.01', '1.0', '0.1', '0.01'), 
         col=c('royalblue2','forestgreen','firebrick3','purple','black','orange'),
         lty=c(1,1,1,2,2,2), lwd=ll, bty='n', cex=1.5)
  legend(x=L$rect$left-L$rect$w-0.01, y=L$rect$top, title='LR Schedule', legend=c('Auto', 'Decr'),
         lty=c(1,2), col='darkgray', lwd=ll, bty='n', cex=1.5)
  
  par(mar=c(5.1,4.1,4.1,2.1))
  if (pdf_bool) {
    dev.off()
  }
  
}

# Rebuttal runs
# Load Mnist Data (see mnist_binary.R)
data.mnist = readRDS("data.mnist.rds")

# Alternative momentum configurations
params_0.2m = momentum_sgd(data.mnist, gamma=0.1, nepoch=2, beta=0.2, batch_size=10, 
                           burnin=0.2, momentum_switch=TRUE, autoLR=TRUE)
params_0.4m = momentum_sgd(data.mnist, gamma=0.1, nepoch=2, beta=0.4, batch_size=10, 
                           burnin=0.2, momentum_switch=TRUE, autoLR=TRUE)
params_0.6m = momentum_sgd(data.mnist, gamma=0.1, nepoch=2, beta=0.6, batch_size=10, 
                           burnin=0.2, momentum_switch=TRUE, autoLR=TRUE)
params_0.8m = momentum_sgd(data.mnist, gamma=0.1, nepoch=2, beta=0.8, batch_size=10, 
                           burnin=0.2, momentum_switch=TRUE, autoLR=TRUE)

plotmom_pdf(params_0.2m, params_0.4m, params_0.6m, params_0.8m, pdf_bool=TRUE)


# Constant SGDM
params_0.001const = momentum_sgd(data.mnist, gamma=0.001, nepoch=2, beta=0.2, batch_size=10, 
                           burnin=0.2, momentum_switch=TRUE, autoLR=FALSE)
params.mnist = readRDS("params.mnist_07-19-19.rds")
params_1.0_notune = params.mnist[[4]]
params_0.1_notune = params.mnist[[5]]
params_0.01_notune= params.mnist[[6]]
plotconst_pdf(params_1.0_notune, params_0.1_notune, params_0.01_notune, params_0.001const, pdf_bool=TRUE)

# Test Stat
#(Using params_0.001const)
plotstat_pdf(params_0.001const, pdf_bool=TRUE)

# LR Curves
params.mnist = readRDS("params.mnist_07-19-19.rds")
params_1.0 = params.mnist[[1]]
params_0.1 = params.mnist[[2]]
params_0.01= params.mnist[[3]]

plotLR_pdf(params_1.0, params_0.1, params_0.01, pdf_bool=FALSE)

# IP vs Mom
params_0.2mIP = momentum_sgd(data.mnist, gamma=0.1, nepoch=1, beta=0.2, batch_size=10, 
                           burnin=0.2, momentum_switch=FALSE, autoLR=FALSE)
params_0.4mIP = momentum_sgd(data.mnist, gamma=0.1, nepoch=1, beta=0.4, batch_size=10, 
                           burnin=0.2, momentum_switch=FALSE, autoLR=FALSE)
params_0.6mIP = momentum_sgd(data.mnist, gamma=0.1, nepoch=1, beta=0.6, batch_size=10, 
                           burnin=0.2, momentum_switch=FALSE, autoLR=FALSE)
params_0.8mIP = momentum_sgd(data.mnist, gamma=0.1, nepoch=1, beta=0.8, batch_size=10, 
                           burnin=0.2, momentum_switch=FALSE, autoLR=FALSE)

plotIPMom_pdf(params_0.2mIP, params_0.4mIP, params_0.6mIP, params_0.8mIP, pdf_bool=TRUE)

# Mom check close stationarity
params_0.4mstat = momentum_sgd(data.mnist, gamma=0.1, nepoch=1, beta=0.4, batch_size=10, 
                             burnin=0.2, momentum_switch=TRUE, autoLR=FALSE)
params_0.6mstat = momentum_sgd(data.mnist, gamma=0.1, nepoch=1, beta=0.6, batch_size=10, 
                             burnin=0.2, momentum_switch=TRUE, autoLR=FALSE)
params_0.8mstat = momentum_sgd(data.mnist, gamma=0.1, nepoch=1, beta=0.8, batch_size=10, 
                             burnin=0.2, momentum_switch=TRUE, autoLR=FALSE)
plotCheckStat_pdf(params_0.4mstat, params_0.6mstat, params_0.8mstat, pdf_bool=TRUE)

# abation mom reduction
params_0.8mAB = momentum_sgd(data.mnist, gamma=0.1, nepoch=2, beta=0.8, batch_size=10, 
                           burnin=0.2, momentum_switch=FALSE, autoLR=TRUE)
plotstat2_pdf(params_0.8mAB, pdf_bool=TRUE)


# Experimental Runs
# keep batch size power of 10
# params_1.0 = momentum_sgd(data.mnist, gamma=1, nepoch=2, beta=0.8, batch_size=10, 
#                       burnin=0.2, momentum_switch=TRUE, autoLR=TRUE)
# 
# params_0.1 = momentum_sgd(data.mnist, gamma=0.1, nepoch=2, beta=0.8, batch_size=10, 
#                       burnin=0.2, momentum_switch=TRUE, autoLR=TRUE)
# 
# params_0.01 = momentum_sgd(data.mnist, gamma=0.01, nepoch=2, beta=0.8, batch_size=10, 
#                       burnin=0.2, momentum_switch=TRUE, autoLR=TRUE)

# no autoLR
# params_1.0_notune = momentum_sgd(data.mnist, gamma=1, nepoch=2, beta=0.8, batch_size=10,
#                       burnin=0.2, momentum_switch=TRUE, autoLR=FALSE)
# 
# params_0.1_notune = momentum_sgd(data.mnist, gamma=0.1, nepoch=2, beta=0.8, batch_size=10,
#                       burnin=0.2, momentum_switch=TRUE, autoLR=FALSE)
# 
# params_0.01_notune = momentum_sgd(data.mnist, gamma=0.01, nepoch=2, beta=0.8, batch_size=10,
#                       burnin=0.2, momentum_switch=TRUE, autoLR=FALSE)

# decr lr
# params_1.0_decr = momentum_sgd(data.mnist, gamma=1, nepoch=2, beta=0.8, batch_size=10,
#                                  burnin=0.2, momentum_switch=TRUE, autoLR=FALSE)
# 
# params_0.1_decr = momentum_sgd(data.mnist, gamma=0.1, nepoch=2, beta=0.8, batch_size=10,
#                                  burnin=0.2, momentum_switch=TRUE, autoLR=FALSE)
# 
# params_0.01_decr = momentum_sgd(data.mnist, gamma=0.01, nepoch=2, beta=0.8, batch_size=10,
#                                   burnin=0.2, momentum_switch=TRUE, autoLR=FALSE)

# saveRDS(list(params_1.0, params_0.1, params_0.01,
#              params_1.0_notune, params_0.1_notune, params_0.01_notune,
#              params_1.0_decr, params_0.1_decr, params_0.01_decr),
#         "params.mnist_0719.rds")


# loads from data.rds
params.mnist = readRDS("params.mnist_07-19-19.rds")
params_1.0 = params.mnist[[1]]
params_0.1 = params.mnist[[2]]
params_0.01= params.mnist[[3]]
params_1.0_notune = params.mnist[[4]]
params_0.1_notune = params.mnist[[5]]
params_0.01_notune= params.mnist[[6]]
params_1.0_decr = params.mnist[[7]]
params_0.1_decr = params.mnist[[8]]
params_0.01_decr= params.mnist[[9]]


# for section 6 plot
# turn pdf_bool=TRUE to save pdf output
plot_pdf(params_1.0, params_0.1, params_0.01,
         params_1.0_notune, params_0.1_notune, params_0.01_notune,
         params_1.0_decr, params_0.1_decr, params_0.01_decr, pdf_bool=FALSE)


