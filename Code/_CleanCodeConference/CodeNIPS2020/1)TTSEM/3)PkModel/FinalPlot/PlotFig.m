function PlotFig1

load data.txt;


len = 1e5+1; 

saem = data(1:30,2);
isaem = data(31:60,2);
vrsaem = data(61:90,2);
fisaem = data(91:120,2);

x = 0:29; 

figure;box on; grid on; 
semilogy(x,isaem,'b-','linewidth',2);hold on; grid on;
plot(x,vrsaem,'k-','linewidth',2);hold on; grid on;
plot(x,fisaem,'g-','linewidth',2);hold on; grid on;  
plot(x,saem,'r-','linewidth',2);hold on; grid on;
set(gca,'fontsize',20);
xlabel('Epoch');
ylabel('|k_a - k_a^*|^2');
legend('iSAEM','fiSAEM','vrSAEM','SAEM')
set(gca,'yminorgrid','off');
set(gca,'xtick',[0:10:20]);
