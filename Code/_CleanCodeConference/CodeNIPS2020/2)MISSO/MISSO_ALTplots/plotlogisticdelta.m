misso10=dlmread(['./Logistic/delta/misso10.txt']);
misso50=dlmread(['./Logistic/delta/misso50.txt']);
mcem=dlmread(['./Logistic/delta/mcem.txt']);
saem=dlmread(['./Logistic/delta/saem.txt']);
misso=dlmread(['./Logistic/delta/misso.txt']);

 
 
figure;
plot(misso10(:,1),misso10(:,2),'k--','linewidth',2);hold on;grid on;
plot(misso50(:,1),misso50(:,2),'r--','linewidth',2);hold on;grid on;
plot(mcem(:,1),mcem(:,2),'g--','linewidth',2);hold on;grid on;
plot(saem(:,1),saem(:,2),'m--','linewidth',2);hold on;grid on;
plot(misso(:,1),misso(:,2),'b--','linewidth',2);hold on;grid on;
 
xlim([0 2100])
ylim([-0.02 0.005])
xlabel('Wallclock (s.)')
ylabel('{\delta_1}')
 
set(gca,'fontsize',20);
legend('MISSO10','MISSO50','MCEM','SAEM','MISSO','Location','southeast');