function PlotFig1

load data.txt;
load data2.txt;

len = 1e5+1; 
isaem = data(1:len,3);
isaemsaga = data( len*1+1:len*1+len,3);
isaemvr = data2(:,4);
IEM = data(len*3+1:len*3+len,3);
EM = data(end-3:end-2,3);
saem = data(end-1:end,3);

x = (0:1e5)/1e5; 

figure;box on; grid on; 
semilogy(x,isaem,'r-','linewidth',2);hold on; grid on;
plot(x(1:10:end),isaemvr,'k-','linewidth',2);hold on; grid on;
plot(x,isaemsaga,'g-','linewidth',2);hold on; grid on;  
plot(x,IEM,'b-','linewidth',2);hold on; grid on;
plot(x([1 end]),EM,'b--','linewidth',2);hold on; grid on;
plot(x([1 end]),saem,'r-.','linewidth',2);hold on; grid on;
set(gca,'fontsize',20);
xlabel('Epoch');
ylabel('|\mu - \mu^*|^2');
%legend('isaem','isamesaga','isaemvr','IEM','EM','saem');
legend('iSAEM','vrSAEM','fiSAEM','iEM','EM','SAEM')
set(gca,'yminorgrid','off');
set(gca,'xtick',[0:0.2:1]);
