data='MNIST';
model='Lenet5';
 

adam=dlmread(['./Lenet5/adam.txt']);
bbb=dlmread(['./Lenet5/bbb.txt']);
misso=dlmread(['./Lenet5/misso.txt']);
momentum=dlmread(['./Lenet5/momentum.txt']);
sag=dlmread(['./Lenet5/sag.txt']);
 
 
xaxis = 1:length(misso(:,2))

figure;
plot(xaxis,momentum(:,2),'k--','linewidth',2);hold on;grid on;
plot(xaxis,adam(:,2),'r--','linewidth',2);hold on;grid on;
plot(xaxis,bbb(:,2),'g--','linewidth',2);hold on;grid on;
plot(xaxis,sag(:,2),'m--','linewidth',2);hold on;grid on;
plot(xaxis,misso(:,2),'b--','linewidth',2);hold on;grid on;
 
xlim([0 100])
xlabel('Epochs')
ylabel('Negated ELBO')
 
text(70,200000,['MNIST + LeNet-5' ],'fontsize',20,'fontweight','bold','color','k');
set(gca,'fontsize',20);
legend('MC-Momentum','MC-Adam','BBB','MC-SAG','MISSO','Location','northeast');

