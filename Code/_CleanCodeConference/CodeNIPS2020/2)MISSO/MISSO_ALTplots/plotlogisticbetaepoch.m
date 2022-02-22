misso10=dlmread(['./Logistic/beta/misso10b.txt']);
misso50=dlmread(['./Logistic/beta/misso50b.txt']);
mcem=dlmread(['./Logistic/beta/mcemb.txt']);
saem=dlmread(['./Logistic/beta/saemb.txt']);
misso=dlmread(['./Logistic/beta/missob.txt']);

 
xaxis = 0:(length(misso(:,2))-1)

figure;
plot(xaxis,misso10(:,2),'k--','linewidth',2);hold on;grid on;
plot(xaxis,misso50(:,2),'r--','linewidth',2);hold on;grid on;
plot(xaxis,mcem(:,2),'g--','linewidth',2);hold on;grid on;
plot(xaxis,saem(:,2),'m--','linewidth',2);hold on;grid on;
plot(xaxis,misso(:,2),'b--','linewidth',2);hold on;grid on;
 
xlim([0 9])
ylim([50 105])
xlabel('Epochs')
ylabel('{\beta_1}')
 
set(gca,'fontsize',20);
legend('MISSO10','MISSO50','MCEM','SAEM','MISSO','Location','northeast');