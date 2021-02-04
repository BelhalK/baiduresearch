library(data.table)
setwd("~/Documents/LargeBatch/code_LB/checkpoint/")

plot_fig <- function(mnist, mnist.noise, savePDF=FALSE, fname, title,
                     ylim=NA, legend=NA) {
  if (savePDF) {
    pdf(fname, width=11, height=8)  
  }
  par(mar=c(5.1,5.1,4.1,2.1))
  plot(mnist$`Valid Acc.`, col='firebrick2', type='l', lwd=5,
       xlab='Epoch', ylab='Test Accuracy (%)', main=title,
       cex.lab=2, cex.axis=2, cex.main=2)
  grid(col='darkgray', lty=1)
  lines(mnist$`Valid Acc.`, col='firebrick2', lwd=5)
  lines(mnist.noise$`Valid Acc.`, col='forestgreen', lwd=7)
  legend(x='bottomright', legend=c("No Noise", "Noise"), col=c('firebrick2', "forestgreen"), lwd=5, lty=1, cex=1.8,
         bty='o', bg='white')
  if (savePDF) {
    dev.off()
  }
  par(mar=c(5.1,4.1,4.1,2.1))
}


#MNIST
mnist.dat = fread('logMnistNet-BS16384-Mom0.5-LR0.1-eta0.0-gamma0.55.txt')
mnistNoise.dat = fread('logMnistNet-BS16384-Mom0.5-LR0.1-eta0.3-gamma0.55.txt')
#plot_fig(mnist.dat, mnistNoise.dat, savePDF=FALSE, 
#         fname="../../Master-Template-ICLR2019/mnistNoise.pdf",
#         title="Gradient Noise Stabilizes Training on MNIST")


# MNIST Full Batch Gradient Descent
mnistGD.dat = fread('logmnist-MnistNet-BS60000-Mom0.5-LR0.01-eta0.0-gamma0.55.txt')
mnistSimSGD.dat = fread('logmnist-MnistNet-BS60000-Mom0.5-LR0.01-eta1.0-gamma0.55.txt')
savePDF=FALSE
fname="../../Master-Template-ICLR2019/mnistSimSGD.pdf"
title="Simulated SGD vs. Full Batch GD on MNIST"
run = TRUE
if (run) {
  if (savePDF) {
    pdf(fname, width=11, height=8)  
  }
  par(mar=c(5.1,5.1,4.1,2.1))
  plot(mnistGD.dat$`Valid Acc.`, col='firebrick2', type='l', lwd=5, ylim=c(10,100),
       xlab='Epoch / Iteration', ylab='Test Accuracy (%)', main=title,
       cex.lab=2, cex.axis=2, cex.main=2)
  grid(col='darkgray', lty=1)
  abline(h=c(30,50,70,90), col='darkgray')
  abline(v=c(10,30,50,70,90), col='darkgray')
  lines(mnistGD.dat$`Valid Acc.`, col='firebrick2', lwd=5)
  lines(mnistSimSGD.dat$`Valid Acc.`, col='forestgreen', lwd=7)
  legend(x='bottomright', legend=c("Full Batch GD", "Simulated SGD"), col=c('firebrick2', "forestgreen"), 
         lwd=5, lty=1, cex=1.8,
              bty='o', bg='white')
  if (savePDF) {
    dev.off()
  }
  par(mar=c(5.1,4.1,4.1,2.1))
}