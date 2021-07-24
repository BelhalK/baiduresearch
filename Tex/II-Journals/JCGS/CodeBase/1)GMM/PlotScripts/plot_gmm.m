function plot_gmm

dir = '../fig/gmm_methods/';

iSAEM = feval('load',[dir 'iSAEM.txt']);
vrTTEM = feval('load',[dir 'vrTTEM.txt']);
fiTTEM = feval('load',[dir 'fiTTEM.txt']);
iEM = feval('load',[dir 'iEM.txt']);
EM = feval('load',[dir 'EM.txt']);
SAEM = feval('load',[dir 'SAEM.txt']);

x = (1:1e5)/1e5; 

figure;
semilogy(x,fiTTEM,'g-','linewidth',2); hold on; grid on;
plot(x(1:10:end),vrTTEM,'r-','linewidth',2);
plot(x,iSAEM,'b-','linewidth',2);
plot(x([1 end]),SAEM,'m-','linewidth',2);
plot(x([1 end]),EM,'k-','linewidth',2);hold on; grid on;
plot(x,iEM,'k--','linewidth',2);hold on; grid on;
 
set(gca,'fontsize',20);
xlabel('Epoch');
ylabel('|\mu - \mu^*|^2');
set(gca,'yminorgrid','off');
set(gca,'ytick',[1e-3 1e-2 1e-1 1e0 1e1]);
set(gca,'xtick',[0:0.2:1]);
legend('fiTTEM','vrTTEM','iSAEM','SAEM','EM','iEM')



