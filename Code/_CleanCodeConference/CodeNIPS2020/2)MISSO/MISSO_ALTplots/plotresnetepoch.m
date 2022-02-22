data='CIFAR';
model='Resnet18';
 
 
adagrad=dlmread(['./Resnet18/adagrad.txt']);
adam=dlmread(['./Resnet18/adam.txt']);
misso=dlmread(['./Resnet18/misso.txt']);
 
xaxis = 1:length(misso(:,2))
 
figure;
plot(xaxis,adagrad(xaxis,2),'k--','linewidth',2);hold on;grid on;
plot(xaxis,adam(xaxis,2),'r--','linewidth',2);hold on;grid on;
plot(xaxis,misso(xaxis,2),'b--','linewidth',2);hold on;grid on;
 
xlim([0 194])
xlabel('Epochs')
ylabel('Negated ELBO')
 
text(70,200000,['CIFAR + RESNET18' ],'fontsize',20,'fontweight','bold','color','k');
set(gca,'fontsize',20);
legend('MC-Adagrad','MC-Adam','MISSO','Location','northeast');
 

