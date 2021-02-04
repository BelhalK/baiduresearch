# Momentum plotting
plotmom_pdf <- function(params1, params2, params3, params4,
                     pdf_bool=FALSE) {
  if (pdf_bool) {
    pdf("RFig2_AltMom.pdf", width=10, height=8)
  }
  
  par(mar=c(5.1,5.1,4.1,2.1))
  
  ll = 2.5
  x = seq(0, 2, length.out=length(params1$out$test_acc))
  plot(x, params1$out$test_acc, col='royalblue2', type='l', ylim=c(0.85, 0.97), lwd=ll,
       cex.main=2, cex.axis=2, cex.lab=2,
       xlab='Epoch', ylab='Test Accuracy', main='Figure 2: Momentum Config on MNIST')
  grid(col='darkgray', lwd=2, lty=1)
  lines(x, params1$out$test_acc, col='royalblue2', lwd=ll)
  lines(x, params2$out$test_acc, col='forestgreen', lwd=ll)
  lines(x, params3$out$test_acc, col='firebrick3', lwd=ll)
  lines(x, params4$out$test_acc, col='orange', lwd=ll)
  L = legend(x='bottomright', title='Initial Momentum', 
             legend=c('0.2', '0.4', '0.6', '0.8'), 
             col=c('royalblue2','forestgreen','firebrick3','orange'),
             lty=c(1,1,1,1), lwd=ll, cex=1.5)
  rect(xleft=L$rect$left, xright=L$rect$left+L$rect$w, 
       ybottom=L$rect$top-L$rect$h, ytop=L$rect$top, col='white', lwd=1)
  L = legend(x='bottomright', title='Initial Momentum', 
             legend=c('0.2', '0.4', '0.6', '0.8'), 
             col=c('royalblue2','forestgreen','firebrick3','orange'),
             lty=c(1,1,1,1), lwd=ll, cex=1.5)
  par(mar=c(5.1,4.1,4.1,2.1))
  if (pdf_bool) {
    dev.off()
  }
  
}

# Const LR
plotconst_pdf <- function(params1, params2, params3, params4,
                        pdf_bool=FALSE) {
  if (pdf_bool) {
    pdf("RFig5_Const.pdf", width=10, height=8)
  }
  
  par(mar=c(5.1,5.1,4.1,2.1))
  
  ll = 5
  x = seq(0, 2, length.out=length(params1$out$test_acc))
  plot(x, params1$out$test_acc, col='royalblue2', type='l', ylim=c(0.85, 0.97), lwd=ll,
       cex.main=2, cex.axis=2, cex.lab=2,
       xlab='Epoch', ylab='Test Accuracy', main='Figure 5: Constant SGDM on MNIST')
  grid(col='darkgray', lwd=2, lty=1)
  lines(x, params1$out$test_acc, col='royalblue2', lwd=ll)
  lines(x, params2$out$test_acc, col='forestgreen', lwd=ll)
  lines(x, params3$out$test_acc, col='firebrick3', lwd=ll)
  lines(x, params4$out$test_acc, col='orange', lwd=ll)
  
  
  L = legend(x='bottomright', title='Learning Rate', 
             legend=c('1.0', '0.1', '0.01', '0.001'), 
             col=c('royalblue2','forestgreen','firebrick3', 'orange'),
             lty=c(1,1,1,1), lwd=ll, bty='o', cex=1.5)
  rect(xleft=L$rect$left, xright=L$rect$left+L$rect$w, 
       ybottom=L$rect$top-L$rect$h, ytop=L$rect$top, col='white', lwd=1)
  L = legend(x='bottomright', title='Learning Rate', 
             legend=c('1.0', '0.1', '0.01', '0.001'), 
             col=c('royalblue2','forestgreen','firebrick3', 'orange'),
             lty=c(1,1,1,1), lwd=ll, bty='o', cex=1.5)
  
  
  
  par(mar=c(5.1,4.1,4.1,2.1))
  if (pdf_bool) {
    dev.off()
  }
  
}

# Test Stat
# test acc is evaluated on intervals, ip evaluated at every interval. Looks like 1:10 ratio
# looking at which(cumsum < 1e-8), see get to zero at 1663, which translates to 166 in test acc
# or 235 and 2347
plotstat_pdf <- function(params, pdf_bool=FALSE) {
  args = params$args
  var  = params$var
  out  = params$out
  par(mar=c(5.1,5.1,4.1,2.1))
  
  if (pdf_bool) {
    pdf("RFig3_Stat.pdf", width=10, height=8)
  }
  ll = 5
  par(mfrow=c(2,1))
  x = seq(0, 2, length.out=length(params$out$test_acc))
  plot(x, out$test_acc, type='l', xlab='Epoch', ylab='Test Acc', 
       main='Figure 3: Test Acc for SGDM with lr=0.001 on MNSIT',
       cex.main=2, cex.axis=2, cex.lab=2, lwd=ll)
  yline = 166/length(params$out$test_acc) * 2
  which(abs(x-yline) < 1e-3)
  abline(v=x[167], col='red', lty=2, lwd=ll)
  legend(x="bottomright", legend=c("Diagnostic Activation"), lty=2, lwd=ll, col='red', cex=1.5)
  
  csum = cumsum(tail(out$ip_all, args$niter_batch*args$nepoch - var$burnin[1]))
  x = seq(0, 2, length.out=length(csum))
  plot(x, csum, type='l', xlab='Epoch', ylab='Test Stat',
       main='Test Statistic for SGDM with lr=0.001 on MNIST',
       cex.main=2, cex.axis=2, cex.lab=2, lwd=ll)
  abline(h=0, lwd=ll)
  # times 167 * 10 from prev plot
  abline(v=x[1670], col='red', lty=2, lwd=ll)
  
  par(mar=c(5.1,4.1,4.1,2.1))
  if (pdf_bool) {
    dev.off()
  }
  
  par(mfrow=c(1,1))
}

# LR Curves
# can re-create from data I already have
plotLR_pdf <- function(params1, params2, params3,
                     pdf_bool=FALSE) {
  if (pdf_bool) {
    pdf("RFig1_LR.pdf", width=10, height=8)
  }
  
  par(mar=c(5.1,5.1,4.1,2.1))
  
  ll = 5
  x = seq(0, 2, length.out=length(params1$out$test_acc))
  plot(x, params1$out$test_acc, col='royalblue2', type='l', ylim=c(0.85, 0.97), lwd=ll,
       cex.main=2, cex.axis=2, cex.lab=2,
       xlab='Epoch', ylab='Test Accuracy', main='Figure 1: LR Reduction on MNIST')
  grid(col='darkgray', lwd=2, lty=1)
  lines(x, params1$out$test_acc, col='royalblue2', lwd=ll)
  lines(x, params2$out$test_acc, col='forestgreen', lwd=ll)
  lines(x, params3$out$test_acc, col='firebrick3', lwd=ll)
  
  yline = (c(params1$var$convg[1],
            params2$var$convg[1],
            params3$var$convg[1]) / params1$args$niter_batch)
  diff = x[2] - x[1]
  abline(v=x[round(yline/diff)], col=c('royalblue2', 'forestgreen', 'firebrick3'),
         lty=2, lwd=ll)
  
  
  L = legend(x='bottomright', title='Initial LR', 
             legend=c('1.0', '0.1', '0.01'), 
             col=c('royalblue2','forestgreen','firebrick3'),
             lty=c(1,1,1,2,2,2), lwd=ll, bty='n', cex=1.5)
  rect(xleft=L$rect$left-L$rect$w-0.02, xright=L$rect$left+L$rect$w, 
       ybottom=L$rect$top-L$rect$h, ytop=L$rect$top, col='white', lwd=1)
  legend(x='bottomright', title='Initial LR', 
         legend=c('1.0', '0.1', '0.01'), 
         col=c('royalblue2','forestgreen','firebrick3'),
         lty=c(1,1,1,2,2,2), lwd=ll, bty='n', cex=1.5)
  legend(x=L$rect$left-L$rect$w-0.01, y=L$rect$top, title='LR Reduction', legend=c('','',''),
         lty=c(2), col=c('royalblue2','forestgreen','firebrick3'), lwd=ll, bty='n', cex=1.5)
  
  par(mar=c(5.1,4.1,4.1,2.1))
  if (pdf_bool) {
    dev.off()
  }
}

# IP vs Mom
plotIPMom_pdf <- function(params1, params2, params3, params4, pdf_bool=FALSE) {
  if (pdf_bool) {
    pdf("RFig8_IPMom.pdf", width=10, height=8)
  }

  args1 = params1$args; var1 = params1$var; out1  = params1$out
  args2 = params2$args; var2 = params2$var; out2  = params2$out
  args3 = params3$args; var3 = params3$var; out3  = params3$out
  args4 = params4$args; var4 = params4$var; out4  = params4$out
  
  csum1 = cumsum(tail(out1$ip_all, args1$niter_batch*args1$nepoch - var1$burnin[1]))
  csum2 = cumsum(tail(out2$ip_all, args2$niter_batch*args2$nepoch - var2$burnin[1]))
  csum3 = cumsum(tail(out3$ip_all, args3$niter_batch*args3$nepoch - var3$burnin[1]))
  csum4 = cumsum(tail(out4$ip_all, args4$niter_batch*args4$nepoch - var4$burnin[1]))
  
  len_csum1 = length(csum1)
  x = seq(0, 2, length.out=len_csum1)
  
  ll = 5
  par(mar=c(5.1,5.1,4.1,2.1))
  
  plot(x, csum1, col='royalblue2', type='l', 
       ylim=c(min(c(csum1,csum2,csum3,csum4)), max(c(csum1,csum2,csum3,csum4))),
       lwd=ll,
       cex.main=2, cex.axis=2, cex.lab=2,
       xlab='Epoch', ylab='Test Stat', main='Figure 8: Test Stat vs Momentum')
  grid(col='darkgray', lwd=2, lty=1)
  lines(x, csum1, col='royalblue2', lwd=ll)
  lines(x, csum2[1:len_csum1], col='forestgreen', lwd=ll)
  lines(x, csum3[1:len_csum1], col='firebrick3', lwd=ll)
  lines(x, csum4[1:len_csum1], col='orange', lwd=ll)
  abline(h=0, lwd=ll)
  L = legend(x='bottomleft', title='Momentum', 
             legend=c('0.2', '0.4', '0.6', '0.8'), 
             col=c('royalblue2','forestgreen','firebrick3','orange'),
             lty=c(1,1,1,1), lwd=ll, cex=1.5)
  rect(xleft=L$rect$left, xright=L$rect$left+L$rect$w, 
       ybottom=L$rect$top-L$rect$h, ytop=L$rect$top, col='white', lwd=1)
  L = legend(x='bottomleft', title='Momentum', 
             legend=c('0.2', '0.4', '0.6', '0.8'), 
             col=c('royalblue2','forestgreen','firebrick3','orange'),
             lty=c(1,1,1,1), lwd=ll, cex=1.5)
  par(mar=c(5.1,4.1,4.1,2.1))
  
  
  par(mfrow=c(1,1))
  if (pdf_bool) {
    dev.off()
  }
}

# Mom check reduce close stationarity.
# can re-create from data I already have
plotCheckStat_pdf <- function(params1, params2, params3,
                       pdf_bool=FALSE) {
  if (pdf_bool) {
    pdf("RFig4_CheckStat.pdf", width=10, height=8)
  }
  
  par(mar=c(5.1,5.1,4.1,2.1))
  
  ll = 2
  x = seq(0, 2, length.out=length(params1$out$test_acc))
  plot(x, params1$out$test_acc, col='royalblue2', type='l', ylim=c(0.85, 0.97), lwd=ll,
       cex.main=2, cex.axis=2, cex.lab=2,
       xlab='Epoch', ylab='Test Accuracy', main='Figure 4: Mom Reduce Close to Stationary MNIST')
  grid(col='darkgray', lwd=2, lty=1)
  lines(x, params1$out$test_acc, col='royalblue2', lwd=ll)
  lines(x, params2$out$test_acc, col='forestgreen', lwd=ll)
  lines(x, params3$out$test_acc, col='firebrick3', lwd=ll)
  
  yline = (c(params1$var$mom_switch,
             params2$var$mom_switch,
             params3$var$mom_switch) / params1$args$niter_batch)
  diff = x[2] - x[1]
  abline(v=x[round(yline/diff)], col=c('royalblue2', 'forestgreen', 'firebrick3'),
         lty=c(2,3,4), lwd=ll+1)
  print(x[round(yline/diff)])
  
  
  L = legend(x='bottomright', title='Mom', 
             legend=c('0.4', '0.6', '0.8'), 
             col=c('royalblue2','forestgreen','firebrick3'),
             lty=c(1,1,1,2,2,2), lwd=ll, bty='n', cex=1.5)
  rect(xleft=L$rect$left-L$rect$w-0.02, xright=L$rect$left+L$rect$w, 
       ybottom=L$rect$top-L$rect$h, ytop=L$rect$top, col='white', lwd=1)
  legend(x='bottomright', title='Mom', 
         legend=c('0.4', '0.6', '0.8'), 
         col=c('royalblue2','forestgreen','firebrick3'),
         lty=c(1,1,1,2,2,2), lwd=ll, bty='n', cex=1.5)
  legend(x=L$rect$left-L$rect$w-0.01, y=L$rect$top, title='Mom Reduce', legend=c('','',''),
         lty=c(2), col=c('royalblue2','forestgreen','firebrick3'), lwd=ll, bty='n', cex=1.5)
  
  par(mar=c(5.1,4.1,4.1,2.1))
  if (pdf_bool) {
    dev.off()
  }
}


# ablation
plotstat2_pdf <- function(params, pdf_bool=FALSE) {
  args = params$args
  var  = params$var
  out  = params$out
  par(mar=c(5.1,5.1,4.1,2.1))
  
  if (pdf_bool) {
    pdf("RFig6_ablation.pdf", width=10, height=8)
  }
  ll = 5
  par(mfrow=c(2,1))
  x = seq(0, 2, length.out=length(params$out$test_acc))
  plot(x, out$test_acc, type='l', xlab='Epoch', ylab='Test Acc', 
       main='Fig 6: High Mom Test Acc & No Mom Red (MNSIT)',
       cex.main=2, cex.axis=2, cex.lab=2, lwd=ll)

  legend(x="bottomright", legend=c("Diagnostic Activation"), lty=2, lwd=ll, col='red', cex=1.5)
  
  csum = cumsum(tail(out$ip_all, args$niter_batch*args$nepoch - var$burnin[1]))
  x = seq(0, 2, length.out=length(csum))
  plot(x, csum, type='l', xlab='Epoch', ylab='Test Stat',
       main='Test Statistic for SGDM on MNIST',
       cex.main=2, cex.axis=2, cex.lab=2, lwd=ll)
  abline(h=0, lwd=ll)

  par(mar=c(5.1,4.1,4.1,2.1))
  if (pdf_bool) {
    dev.off()
  }
  
  par(mfrow=c(1,1))
}
