misso10=dlmread(['./Logistic/delta/misso10.txt']);
misso50=dlmread(['./Logistic/delta/misso50.txt']);
mcem=dlmread(['./Logistic/delta/mcem.txt']);
saem=dlmread(['./Logistic/delta/saem.txt']);
misso=dlmread(['./Logistic/delta/misso.txt']);

 
xaxis = 0:(length(misso(:,2))-1)
figure;
plot(xaxis,misso10(:,2),'k--','linewidth',2);hold on;grid on;
plot(xaxis,misso50(:,2),'r--','linewidth',2);hold on;grid on;
plot(xaxis,mcem(:,2),'g--','linewidth',2);hold on;grid on;
plot(xaxis,saem(:,2),'m--','linewidth',2);hold on;grid on;
plot(xaxis,misso(:,2),'b--','linewidth',2);hold on;grid on;
 
xlim([0 9])
ylim([-0.02 0.005])
xlabel('Epochs')
ylabel('{\delta_1}')
 
set(gca,'fontsize',20);
legend('MISSO10','MISSO50','MCEM','SAEM','MISSO','Location','southeast');