dir = 'pk_methods/';
x = 0:30;

fiTTEM = feval('load',[dir 'fiTTEM.txt']);
vrTTEM = feval('load',[dir 'vrTTEM.txt']);
iSAEM = feval('load',[dir 'iSAEM.txt']);
SAEM = feval('load',[dir 'SAEM.txt']);

figure;
semilogy(x,fiTTEM,'r-','linewidth',2); hold on; grid on;
plot(x,vrTTEM,'k-','linewidth',2);
plot(x,iSAEM,'b-','linewidth',2);
plot(x,SAEM,'m-','linewidth',2);