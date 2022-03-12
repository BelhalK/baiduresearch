function plot_SNLI

folder = 'SNLI/biLSTM/';textbody = 'SNLI: biLSTM';

x = 1:500;

adabound = importdata([folder 'adabound_train.txt']);
adam = importdata([folder 'adam_train.txt']);
rmsprop = importdata([folder 'rmsprop_train.txt']);
sagd = importdata([folder 'sagd_train.txt']);

figure;box on; grid on; 
plot(x,adabound,'b-','linewidth',1.5);hold on; grid on;
plot(x,adam,'g-','linewidth',1.5);hold on; grid on;  
plot(x,rmsprop,'k-','linewidth',1.5);hold on; grid on;  
plot(x,sagd,'r-','linewidth',1.5);hold on; grid on;  

set(gca,'fontsize',20);
xlabel('Epoch');
ylabel('Train Perplexity');
legend('AdaBound','Adam','RMSprop','SAGD','location','northeast');
set(gca,'xminorgrid','off');
set(gca,'xlim',[0 500],'xtick',0:100:500);
set(gca,'ylim',[0 20]);
text(20,16,textbody,'fontsize',20,'color','r','fontweight','bold');


adabound = importdata([folder 'adabound_test.txt']);
adam = importdata([folder 'adam_test.txt']);
rmsprop = importdata([folder 'rmsprop_test.txt']);
sagd = importdata([folder 'sagd_test.txt']);



figure;box on; grid on; 
plot(x,adabound,'b-','linewidth',1.5);hold on; grid on;
plot(x,adam,'g-','linewidth',1.5);hold on; grid on;  
plot(x,rmsprop,'k-','linewidth',1.5);hold on; grid on;  
plot(x,sagd,'r-','linewidth',1.5);hold on; grid on;  

set(gca,'fontsize',20);
xlabel('Epoch');
ylabel('Test Perplexity');
legend('AdaBound','Adam','RMSprop','SAGD','location','northeast');
set(gca,'xminorgrid','off');
set(gca,'xlim',[0 500],'xtick',0:100:500);
set(gca,'ylim',[5 20]);
text(20,16,textbody,'fontsize',20,'color','r','fontweight','bold');
