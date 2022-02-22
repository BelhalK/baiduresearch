data='MNIST';
model='Lenet5';
 
 
adam=dlmread(['./Lenet5/adam.txt']);
bbb=dlmread(['./Lenet5/bbb.txt']);
misso=dlmread(['./Lenet5/misso.txt']);
momentum=dlmread(['./Lenet5/momentum.txt']);
sag=dlmread(['./Lenet5/sag.txt']);
 
 
figure;
plot(momentum(:,1),momentum(:,2),'k--','linewidth',2);hold on;grid on;
plot(adam(:,1),adam(:,2),'r--','linewidth',2);hold on;grid on;
plot(bbb(:,1),bbb(:,2),'g--','linewidth',2);hold on;grid on;
plot(sag(:,1),sag(:,2),'m--','linewidth',2);hold on;grid on;
plot(misso(:,1),misso(:,2),'b--','linewidth',2);hold on;grid on;
 
xlim([0 31000])
xlabel('Wallclock (s.)')
ylabel('Negated ELBO')
 
text(6000,200000,['MNIST + LeNet-5' ],'fontsize',20,'fontweight','bold','color','k');
set(gca,'fontsize',20);
legend('MC-Momentum','MC-Adam','BBB','MC-SAG','MISSO','Location','northeast');