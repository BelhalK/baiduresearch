function plot_pk

dir = '../fig/pk_methods/';
x = 0:30;

fiTTEM = feval('load',[dir 'fiTTEM.txt']);
vrTTEM = feval('load',[dir 'vrTTEM.txt']);
iSAEM = feval('load',[dir 'iSAEM.txt']);
SAEM = feval('load',[dir 'SAEM.txt']);

figure;
semilogy(x,fiTTEM,'g-','linewidth',2); hold on; grid on;
plot(x,vrTTEM,'r-','linewidth',2);
plot(x,iSAEM,'b-','linewidth',2);
plot(x,SAEM,'m-','linewidth',2);
set(gca,'fontsize',20);
xlabel('Epoch');
ylabel('|ka - ka^*|^2');
set(gca,'yminorgrid','off');
set(gca,'ytick',[1e-2 1e-1 1e0 4]);
legend('fiTTEM','vrTTEM','iSAEM','SAEM');

