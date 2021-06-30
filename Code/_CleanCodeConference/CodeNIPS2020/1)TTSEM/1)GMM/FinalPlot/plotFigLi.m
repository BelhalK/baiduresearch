dir = 'gmm_methods/';

iSAEM = feval('load',[dir 'iSAEM.txt']);
vrTTEM = feval('load',[dir 'vrTTEM.txt']);
fiTTEM = feval('load',[dir 'fiTTEM.txt']);
iEM = feval('load',[dir 'iEM.txt']);
EM = feval('load',[dir 'EM.txt']);
SAEM = feval('load',[dir 'SAEM.txt']);

x = (1:1e5)/1e5; 

semilogy(x,fiTTEM,'r-','linewidth',2); hold on; grid on;
plot(x(1:10:end),vrTTEM,'k-','linewidth',2);
plot(x,iSAEM,'b-','linewidth',2);
plot(x([1 end]),SAEM,'m-','linewidth',2);
plot(x([1 end]),EM,'b--','linewidth',2);hold on; grid on;
plot(x,iEM,'r-.','linewidth',2);hold on; grid on;
 
legend('fiTTEM','vrTTEM','iSAEM','SAEM','EM','iEM')