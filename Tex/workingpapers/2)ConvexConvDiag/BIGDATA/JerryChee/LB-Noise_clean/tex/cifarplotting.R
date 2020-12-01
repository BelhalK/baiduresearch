library(data.table)
setwd("~/Documents/LargeBatch/code_LB/checkpoint/")

cifar.8192 = fread("log150-cifar10-EResNet18-BS8192-Mom0.9-LR0.1-eta0.0-gamma0.55.txt")
#cifar.8192.noise = fread("log150-cifar10-EResNet18-BS8192-Mom0.9-LR0.05-eta0.01-gamma0.55.txt")
cifar.8192.noise = fread("logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.1-eta0.005-gamma0.55.txt")
cifar.4096 = fread("log150-cifar10-EResNet18-BS4096-Mom0.9-LR0.1-eta0.0-gamma0.55.txt")
cifar.4096.noise = fread("log150-cifar10-EResNet18-BS4096-Mom0.9-LR0.05-eta0.01-gamma0.55.txt")
cifar.2048 = fread("log150-cifar10-EResNet18-BS2048-Mom0.9-LR0.1-eta0.0-gamma0.55.txt")
cifar.2048.noise = fread("log150-cifar10-EResNet18-BS2048-Mom0.9-LR0.05-eta0.01-gamma0.55.txt")



plot_cifar <- function(cifar, cifar.noise) {
  plot(cifar$`Valid Acc.`, type='l')
  lines(cifar.noise$`Valid Acc.`, col='blue')
  grid()
  abline(v=20)
}

plot_cifar(cifar.8192, cifar.8192.noise)

#plot_cifar(cifar.4096, cifar.4096.noise)

#plot_cifar(cifar.2048, cifar.2048.noise)
