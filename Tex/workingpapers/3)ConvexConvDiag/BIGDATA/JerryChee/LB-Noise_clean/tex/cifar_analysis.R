#setwd("Documents/LargeBatch/code_LB/")
library(data.table)

plot_v0.5 <- function(dat0, dat1, dat2, dat3, dat4, dat5, title, legend.names, legend.title) {
  plot(dat0$`Valid Acc.`, type='l', main=title, xlab='Epoch', ylab='Test Accuracy')
  lines(dat1$`Valid Acc.`, col='orange')
  lines(dat2$`Valid Acc.`, col='green')
  lines(dat3$`Valid Acc.`, col='blue')
  lines(dat4$`Valid Acc.`, col='red')
  lines(dat5$`Valid Acc.`, col='purple')
  grid()
  legend(x="bottomright", legend=legend.names, title=legend.title, 
         col=c("black","orange", "green","blue","red","purple"), lty=1)
}

plot_v1.5 <- function(ref, exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8, title, legend.ref) {
  # For plotting comparison for a given BS, LR
  plot(ref$`Valid Acc.`, type='l', main=title, xlab='Epoch', ylab='Test Accuracy')
  lines(exp1$`Valid Acc.`, col='orange')
  lines(exp2$`Valid Acc.`, col='green')
  lines(exp3$`Valid Acc.`, col='blue')
  lines(exp4$`Valid Acc.`, col='red')
  lines(exp5$`Valid Acc.`, col='purple')
  lines(exp6$`Valid Acc.`, col='black', lty=2)
  lines(exp7$`Valid Acc.`, col='orange', lty=2)
  lines(exp8$`Valid Acc.`, col='green', lty=2)
  lines(ref$`Valid Acc.`)
  grid()
  legend(x="bottomright", legend=c(legend.ref, '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.005', '0.01', '0.1'), 
         title="eta", col=c('black','orange','green','blue','red','purple','black','orange','green'), 
         lty=c(1,1,1,1,1,1,2,2,2), bty='o', bg='white')
}

plot_v0 <- function(dat1, dat2, dat3, dat4, dat5, title, legend.names, legend.title) {
  plot(dat1$`Valid Acc.`, type='l', main=title, xlab='Epoch', ylab='Test Accuracy')
  lines(dat2$`Valid Acc.`, col='green')
  lines(dat3$`Valid Acc.`, col='blue')
  lines(dat4$`Valid Acc.`, col='red')
  lines(dat5$`Valid Acc.`, col='purple')
  grid()
  legend(x="bottomright", legend=legend.names, title=legend.title, 
         col=c("black","green","blue","red","purple"), lty=1)
}

plot_v1 <- function(ref, exp1, exp2, exp3, title, legend.ref) {
  # For plotting comparison for a given BS, LR
  plot(ref$`Valid Acc.`, type='l', main=title, xlab='Epoch', ylab='Test Accuracy')
  lines(exp1$`Valid Acc.`, col='orange')
  lines(exp2$`Valid Acc.`, col='green')
  lines(exp3$`Valid Acc.`, col='blue')
  lines(ref$`Valid Acc.`)
  grid()
  legend(x="bottomright", legend=c(legend.ref, '0.005', '0.01', '0.1'), 
         title="eta", col=c('black','orange','green','blue'), 
         lty=c(1), bty='o', bg='white')
}


# Just LR scaling
cifar.8192.05.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.05-eta0.0-gamma0.55.txt")
cifar.8192.1.dat = fread("checkpoint/log150-cifar10-EResNet18-BS8192-Mom0.9-LR0.1-eta0.0-gamma0.55.txt")
cifar.8192.2.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.2-eta0.0-gamma0.55.txt")
cifar.8192.3.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.3-eta0.0-gamma0.55.txt")
cifar.8192.4.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.4-eta0.0-gamma0.55.txt")
cifar.8192.5.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.5-eta0.0-gamma0.55.txt")

# LR=0.05 decreasing noise scaling
cifar.8192.05.eta1e7.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.05-eta1e-07-gamma0.55.txt")
cifar.8192.05.eta1e6.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.05-eta1e-06-gamma0.55.txt")
cifar.8192.05.eta1e5.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.05-eta1e-05-gamma0.55.txt")
cifar.8192.05.eta1e4.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.05-eta0.0001-gamma0.55.txt")
cifar.8192.05.eta1e3.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.05-eta0.001-gamma0.55.txt")
cifar.8192.05.eta005.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.05-eta0.005-gamma0.55.txt")
cifar.8192.05.eta01.dat = fread("checkpoint/log150-cifar10-EResNet18-BS8192-Mom0.9-LR0.05-eta0.01-gamma0.55.txt")
cifar.8192.05.eta1.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.05-eta0.1-gamma0.55.txt")

# LR=0.05 and constant noise scaling
cifar.8192.05.eta1e7.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.05-eta1e-07-gamma0.0.txt")
cifar.8192.05.eta1e6.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.05-eta1e-06-gamma0.0.txt")
cifar.8192.05.eta1e5.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.05-eta1e-05-gamma0.0.txt")
cifar.8192.05.eta1e4.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.05-eta0.0001-gamma0.0.txt")
cifar.8192.05.eta1e3.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.05-eta0.001-gamma0.0.txt")
cifar.8192.05.eta005.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.05-eta0.005-gamma0.0.txt")
cifar.8192.05.eta01.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.05-eta0.01-gamma0.0.txt")
cifar.8192.05.eta1.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.05-eta0.1-gamma0.0.txt")

# LR=0.1 decreasing noise scaling
cifar.8192.1.eta1e7.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.1-eta1e-07-gamma0.55.txt")
cifar.8192.1.eta1e6.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.1-eta1e-06-gamma0.55.txt")
cifar.8192.1.eta1e5.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.1-eta1e-05-gamma0.55.txt")
cifar.8192.1.eta1e4.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.1-eta0.0001-gamma0.55.txt")
cifar.8192.1.eta1e3.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.1-eta0.001-gamma0.55.txt")
cifar.8192.1.eta005.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.1-eta0.005-gamma0.55.txt")
cifar.8192.1.eta01.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.1-eta0.01-gamma0.55.txt")
cifar.8192.1.eta1.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.1-eta0.1-gamma0.55.txt")

# LR=0.1 and constant noise scaling
cifar.8192.1.eta1e7.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.1-eta1e-07-gamma0.0.txt")
cifar.8192.1.eta1e6.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.1-eta1e-06-gamma0.0.txt")
cifar.8192.1.eta1e5.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.1-eta1e-05-gamma0.0.txt")
cifar.8192.1.eta1e4.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.1-eta0.0001-gamma0.0.txt")
cifar.8192.1.eta1e3.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.1-eta0.001-gamma0.0.txt")
cifar.8192.1.eta005.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.1-eta0.005-gamma0.0.txt")
cifar.8192.1.eta01.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.1-eta0.01-gamma0.0.txt")
cifar.8192.1.eta1.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.1-eta0.1-gamma0.0.txt")

# LR=0.2 and decreasing noise scaling
cifar.8192.2.eta1e7.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.2-eta1e-07-gamma0.55.txt")
cifar.8192.2.eta1e6.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.2-eta1e-06-gamma0.55.txt")
cifar.8192.2.eta1e5.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.2-eta1e-05-gamma0.55.txt")
cifar.8192.2.eta1e4.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.2-eta0.0001-gamma0.55.txt")
cifar.8192.2.eta1e3.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.2-eta0.001-gamma0.55.txt")
cifar.8192.2.eta005.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.2-eta0.005-gamma0.55.txt")
cifar.8192.2.eta01.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.2-eta0.01-gamma0.55.txt")
cifar.8192.2.eta1.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.2-eta0.1-gamma0.55.txt")

# LR=0.2 and constant noise scaling
cifar.8192.2.eta1e7.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.2-eta1e-07-gamma0.0.txt")
cifar.8192.2.eta1e6.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.2-eta1e-06-gamma0.0.txt")
cifar.8192.2.eta1e5.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.2-eta1e-05-gamma0.0.txt")
cifar.8192.2.eta1e4.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.2-eta0.0001-gamma0.0.txt")
cifar.8192.2.eta1e3.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.2-eta0.001-gamma0.0.txt")
cifar.8192.2.eta005.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.2-eta0.005-gamma0.0.txt")
cifar.8192.2.eta01.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.2-eta0.01-gamma0.0.txt")
cifar.8192.2.eta1.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.2-eta0.1-gamma0.0.txt")

# LR=0.3 and decreasing noise scaling
cifar.8192.3.eta1e7.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.3-eta1e-07-gamma0.55.txt")
cifar.8192.3.eta1e6.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.3-eta1e-06-gamma0.55.txt")
cifar.8192.3.eta1e5.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.3-eta1e-05-gamma0.55.txt")
cifar.8192.3.eta1e4.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.3-eta0.0001-gamma0.55.txt")
cifar.8192.3.eta1e3.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.3-eta0.001-gamma0.55.txt")
cifar.8192.3.eta005.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.3-eta0.005-gamma0.55.txt")
cifar.8192.3.eta01.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.3-eta0.01-gamma0.55.txt")
cifar.8192.3.eta1.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.3-eta0.1-gamma0.55.txt")

# LR=0.3 and constant noise scaling
cifar.8192.3.eta1e7.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.3-eta1e-07-gamma0.0.txt")
cifar.8192.3.eta1e6.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.3-eta1e-06-gamma0.0.txt")
cifar.8192.3.eta1e5.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.3-eta1e-05-gamma0.0.txt")
cifar.8192.3.eta1e4.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.3-eta0.0001-gamma0.0.txt")
cifar.8192.3.eta1e3.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.3-eta0.001-gamma0.0.txt")
cifar.8192.3.eta005.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.3-eta0.005-gamma0.0.txt")
cifar.8192.3.eta01.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.3-eta0.01-gamma0.0.txt")
cifar.8192.3.eta1.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.3-eta0.1-gamma0.0.txt")

# LR=0.4 and decreasing noise scaling
cifar.8192.4.eta1e7.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.4-eta1e-07-gamma0.55.txt")
cifar.8192.4.eta1e6.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.4-eta1e-06-gamma0.55.txt")
cifar.8192.4.eta1e5.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.4-eta1e-05-gamma0.55.txt")
cifar.8192.4.eta1e4.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.4-eta0.0001-gamma0.55.txt")
cifar.8192.4.eta1e3.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.4-eta0.001-gamma0.55.txt")
cifar.8192.4.eta005.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.4-eta0.005-gamma0.55.txt")
cifar.8192.4.eta01.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.4-eta0.01-gamma0.55.txt")
cifar.8192.4.eta1.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.4-eta0.1-gamma0.55.txt")

# LR=0.4 and constant noise scaling
cifar.8192.4.eta1e7.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.4-eta1e-07-gamma0.0.txt")
cifar.8192.4.eta1e6.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.4-eta1e-06-gamma0.0.txt")
cifar.8192.4.eta1e5.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.4-eta1e-05-gamma0.0.txt")
cifar.8192.4.eta1e4.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.4-eta0.0001-gamma0.0.txt")
cifar.8192.4.eta1e3.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.4-eta0.001-gamma0.0.txt")
cifar.8192.4.eta005.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.4-eta0.005-gamma0.0.txt")
cifar.8192.4.eta01.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.4-eta0.01-gamma0.0.txt")
cifar.8192.4.eta1.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.4-eta0.1-gamma0.0.txt")

# LR=0.5 and decreasing noise scaling
cifar.8192.5.eta1e7.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.5-eta1e-07-gamma0.55.txt")
cifar.8192.5.eta1e6.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.5-eta1e-06-gamma0.55.txt")
cifar.8192.5.eta1e5.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.5-eta1e-05-gamma0.55.txt")
cifar.8192.5.eta1e4.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.5-eta0.0001-gamma0.55.txt")
cifar.8192.5.eta1e3.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.5-eta0.001-gamma0.55.txt")
cifar.8192.5.eta005.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.5-eta0.005-gamma0.55.txt")
cifar.8192.5.eta01.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.5-eta0.01-gamma0.55.txt")
cifar.8192.5.eta1.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.5-eta0.1-gamma0.55.txt")

# LR=0.5 and constant noise scaling
cifar.8192.5.eta1e7.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.5-eta1e-07-gamma0.0.txt")
cifar.8192.5.eta1e6.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.5-eta1e-06-gamma0.0.txt")
cifar.8192.5.eta1e5.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.5-eta1e-05-gamma0.0.txt")
cifar.8192.5.eta1e4.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.5-eta0.0001-gamma0.0.txt")
cifar.8192.5.eta1e3.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.5-eta0.001-gamma0.0.txt")
#cifar.8192.5.eta005.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.5-eta0.005-gamma0.0.txt")
#cifar.8192.5.eta01.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.5-eta0.01-gamma0.0.txt")
#cifar.8192.5.eta1.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS8192-Mom0.9-LR0.5-eta0.1-gamma0.0.txt")

# BS = 4096 ===================================================================
# Just LR scaling
cifar.4096.1.dat = fread("checkpoint/log150-cifar10-EResNet18-BS2048-Mom0.9-LR0.1-eta0.0-gamma0.55.txt")
cifar.4096.2.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.2-eta0.0-gamma0.55.txt")
cifar.4096.3.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.3-eta0.0-gamma0.55.txt")
cifar.4096.4.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.4-eta0.0-gamma0.55.txt")
cifar.4096.5.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.5-eta0.0-gamma0.55.txt")

# LR=0.1 decreasing noise scaling
cifar.4096.1.eta005.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.1-eta0.005-gamma0.55.txt")
cifar.4096.1.eta01.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.1-eta0.01-gamma0.55.txt")
cifar.4096.1.eta1.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.1-eta0.1-gamma0.55.txt")

# LR=0.05 decreasing noise scaling
cifar.4096.05.eta005.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.05-eta0.005-gamma0.55.txt")
cifar.4096.05.eta01.dat = fread("checkpoint/log150-cifar10-EResNet18-BS4096-Mom0.9-LR0.05-eta0.01-gamma0.55.txt")
cifar.4096.05.eta1.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.05-eta0.1-gamma0.55.txt")

# LR=0.05 and constant noise scaling
cifar.4096.05.eta005.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.05-eta0.005-gamma0.0.txt")
cifar.4096.05.eta01.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.05-eta0.01-gamma0.0.txt")
cifar.4096.05.eta1.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.05-eta0.1-gamma0.0.txt")

# LR=0.1 and constant noise scaling
cifar.4096.1.eta005.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.1-eta0.005-gamma0.0.txt")
cifar.4096.1.eta01.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.1-eta0.01-gamma0.0.txt")
cifar.4096.1.eta1.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.1-eta0.1-gamma0.0.txt")

# LR=0.2 and decreasing noise scaling
cifar.4096.2.eta005.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.2-eta0.005-gamma0.55.txt")
cifar.4096.2.eta01.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.2-eta0.01-gamma0.55.txt")
cifar.4096.2.eta1.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.2-eta0.1-gamma0.55.txt")

# LR=0.3 and decreasing noise scaling
cifar.4096.3.eta005.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.3-eta0.005-gamma0.55.txt")
cifar.4096.3.eta01.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.3-eta0.01-gamma0.55.txt")
cifar.4096.3.eta1.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.3-eta0.1-gamma0.55.txt")

# LR=0.2 and constant noise scaling
cifar.4096.2.eta005.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.2-eta0.005-gamma0.0.txt")
cifar.4096.2.eta01.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.2-eta0.01-gamma0.0.txt")
cifar.4096.2.eta1.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.2-eta0.1-gamma0.0.txt")

# LR=0.3 and constant noise scaling
cifar.4096.3.eta005.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.3-eta0.005-gamma0.0.txt")
cifar.4096.3.eta01.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.3-eta0.01-gamma0.0.txt")
cifar.4096.3.eta1.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.3-eta0.1-gamma0.0.txt")

# LR=0.4 and decreasing noise scaling
cifar.4096.4.eta005.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.4-eta0.005-gamma0.55.txt")
cifar.4096.4.eta01.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.4-eta0.01-gamma0.55.txt")
cifar.4096.4.eta1.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.4-eta0.1-gamma0.55.txt")

# LR=0.5 and decreasing noise scaling
cifar.4096.5.eta005.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.5-eta0.005-gamma0.55.txt")
cifar.4096.5.eta01.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.5-eta0.01-gamma0.55.txt")
cifar.4096.5.eta1.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.5-eta0.1-gamma0.55.txt")

# LR=0.4 and constant noise scaling
#cifar.4096.4.eta005.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.4-eta0.005")
#cifar.4096.4.eta01.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.4-eta0.01-gamma0.0.txt")
#cifar.4096.4.eta1.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.4-eta0.1-gamma0.0.txt")

# LR=0.5 and constant noise scaling
#cifar.4096.5.eta005.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.5-eta0.005-gamma0.0.txt")
#cifar.4096.5.eta01.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.5-eta0.01-gamma0.0.txt")
#cifar.4096.5.eta1.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS4096-Mom0.9-LR0.5-eta0.1-gamma0.0.txt")

# BS = 2048 ===================================================================
# Just LR scaling
cifar.2048.1.dat = fread("checkpoint/log150-cifar10-EResNet18-BS2048-Mom0.9-LR0.1-eta0.0-gamma0.55.txt")
cifar.2048.2.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.2-eta0.0-gamma0.55.txt")
cifar.2048.3.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.3-eta0.0-gamma0.55.txt")
cifar.2048.4.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.4-eta0.0-gamma0.55.txt")
cifar.2048.5.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.5-eta0.0-gamma0.55.txt")

# LR=0.1 decreasing noise scaling
cifar.2048.1.eta005.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.1-eta0.005-gamma0.55.txt")
cifar.2048.1.eta01.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.1-eta0.01-gamma0.55.txt")
cifar.2048.1.eta1.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.1-eta0.1-gamma0.55.txt")

# LR=0.05 decreasing noise scaling
cifar.2048.05.eta005.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.05-eta0.005-gamma0.55.txt")
cifar.2048.05.eta01.dat = fread("checkpoint/log150-cifar10-EResNet18-BS2048-Mom0.9-LR0.05-eta0.01-gamma0.55.txt")
cifar.2048.05.eta1.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.05-eta0.1-gamma0.55.txt")

# LR=0.05 and constant noise scaling
cifar.2048.05.eta005.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.05-eta0.005-gamma0.0.txt")
cifar.2048.05.eta01.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.05-eta0.01-gamma0.0.txt")
cifar.2048.05.eta1.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.05-eta0.1-gamma0.0.txt")

# LR=0.1 and constant noise scaling
cifar.2048.1.eta005.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.1-eta0.005-gamma0.0.txt")
cifar.2048.1.eta01.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.1-eta0.01-gamma0.0.txt")
cifar.2048.1.eta1.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.1-eta0.1-gamma0.0.txt")

# LR=0.2 and decreasing noise scaling
cifar.2048.2.eta005.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.2-eta0.005-gamma0.55.txt")
cifar.2048.2.eta01.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.2-eta0.01-gamma0.55.txt")
cifar.2048.2.eta1.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.2-eta0.1-gamma0.55.txt")

# LR=0.3 and decreasing noise scaling
cifar.2048.3.eta005.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.3-eta0.005-gamma0.55.txt")
cifar.2048.3.eta01.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.3-eta0.01-gamma0.55.txt")
cifar.2048.3.eta1.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.3-eta0.1-gamma0.55.txt")

# LR=0.2 and constant noise scaling
cifar.2048.2.eta005.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.2-eta0.005-gamma0.0.txt")
cifar.2048.2.eta01.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.2-eta0.01-gamma0.0.txt")
cifar.2048.2.eta1.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.2-eta0.1-gamma0.0.txt")

# LR=0.3 and constant noise scaling
#cifar.2048.3.eta005.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.3-eta0.005-gamma0.0.txt")
#cifar.2048.3.eta01.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.3-eta0.01-gamma0.0.txt")
#cifar.2048.3.eta1.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.3-eta0.1-gamma0.0.txt")

# LR=0.4 and decreasing noise scaling
#cifar.2048.4.eta005.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.4-eta0.005-gamma0.55.txt")
#cifar.2048.4.eta01.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.4-eta0.01-gamma0.55.txt")
#cifar.2048.4.eta1.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.4-eta0.1-gamma0.55.txt")

# LR=0.5 and decreasing noise scaling
#cifar.2048.5.eta005.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.5-eta0.005-gamma0.55.txt")
#cifar.2048.5.eta01.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.5-eta0.01-gamma0.55.txt")
#cifar.2048.5.eta1.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.5-eta0.1-gamma0.55.txt")

# LR=0.4 and constant noise scaling
#cifar.2048.4.eta005.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.4-eta0.005")
#cifar.2048.4.eta01.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.4-eta0.01-gamma0.0.txt")
#cifar.2048.4.eta1.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.4-eta0.1-gamma0.0.txt")

# LR=0.5 and constant noise scaling
#cifar.2048.5.eta005.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.5-eta0.005-gamma0.0.txt")
#cifar.2048.5.eta01.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.5-eta0.01-gamma0.0.txt")
#cifar.2048.5.eta1.gam0.dat = fread("checkpoint/logcifar10-ResNet18-E150-BS2048-Mom0.9-LR0.5-eta0.1-gamma0.0.txt")