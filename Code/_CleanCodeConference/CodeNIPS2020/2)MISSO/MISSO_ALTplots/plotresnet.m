data='CIFAR';
model='Resnet18';
 
 
adagrad=dlmread(['./Resnet18/adagrad.txt']);
adam=dlmread(['./Resnet18/adam.txt']);
misso=dlmread(['./Resnet18/misso.txt']);
 
 
figure;
plot(adagrad(:,1),adagrad(:,2),'k--','linewidth',2);hold on;grid on;
plot(adam(:,1),adam(:,2),'r--','linewidth',2);hold on;grid on;
plot(misso(:,1),misso(:,2),'b--','linewidth',2);hold on;grid on;
 
xlim([0 31000])
xlabel('Wallclock (s.)')
ylabel('Negated ELBO')
 
text(6000,200000,['CIFAR + RESNET18' ],'fontsize',20,'fontweight','bold','color','k');
set(gca,'fontsize',20);
legend('MC-Adagrad','MC-Adam','MISSO','Location','northeast');
