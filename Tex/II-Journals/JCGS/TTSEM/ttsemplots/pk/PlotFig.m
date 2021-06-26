function PlotFig1

load data.txt;


len = 1e5+1; 

saem = data(2:32,2);
isaem = data(33:63,2);
vrsaem = data(64:94,2);
fisaem= data(95:125,2);

x = 0:30; 

figure;box on; grid on; 
semilogy(x,isaem,'b-','linewidth',2);hold on; grid on;
plot(x,fisaem,'g-','linewidth',2);hold on; grid on;  
plot(x,vrsaem,'k-','linewidth',2);hold on; grid on;
plot(x,saem,'r-','linewidth',2);hold on; grid on;
set(gca,'fontsize',20);
xlabel('Epoch');
ylabel('|ka - ka^*|^2');
legend('iSAEM','fiTTEM','vrTTEM','SAEM')
set(gca,'yminorgrid','off');
set(gca,'xtick',[0:10:30]);
