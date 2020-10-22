function plot_CIFAR

folder = 'PTB/2layer/';textbody = 'PTB: 2LSTM';
%folder = 'PTB/3layer/';textbody = 'PTB: 3LSTM';

x = 1:500

adabound = importdata([folder 'adabound_train.txt']);
adam = importdata([folder 'adam_train.txt']);
rmsprop = importdata([folder 'rmsprop_train.txt']);
sgd = importdata([folder 'sgd_train.txt']);
sagd = importdata([folder 'sagd_train.txt']);

figure;box on; grid on; 
semilogx(x,adabound,'b-','linewidth',1.5);hold on; grid on;
plot(x,adam,'g-','linewidth',1.5);hold on; grid on;  
plot(x,rmsprop,'k-','linewidth',1.5);hold on; grid on;  
plot(x,sgd,'m-','linewidth',1.5);hold on; grid on;  
plot(x,sagd,'r-','linewidth',1.5);hold on; grid on;  
ylim([30 200])

set(gca,'fontsize',20);
xlabel('Epoch');
ylabel('Train Perplexity');
legend('AdaBound','Adam','RMSprop','SGD','SAGD','location','northeast');
set(gca,'xminorgrid','off');
text(-35,1.3,textbody,'fontsize',20,'color','r','fontweight','bold');


adabound = importdata([folder 'adabound_test.txt']);
adam = importdata([folder 'adam_test.txt']);
rmsprop = importdata([folder 'rmsprop_test.txt']);
sgd = importdata([folder 'sgd_test.txt']);
sagd = importdata([folder 'sagd_test.txt']);



figure;box on; grid on; 
semilogx(x,adabound,'b-','linewidth',1.5);hold on; grid on;
plot(x,adam,'g-','linewidth',1.5);hold on; grid on;  
plot(x,rmsprop,'k-','linewidth',1.5);hold on; grid on;  
plot(x,sgd,'m-','linewidth',1.5);hold on; grid on;  
plot(x,sagd,'r-','linewidth',1.5);hold on; grid on;  
ylim([30 200])

set(gca,'fontsize',20);
xlabel('Epoch');
ylabel('Test Perplexity');
legend('AdaBound','Adam','RMSprop','SGD','SAGD','location','southeast');
set(gca,'xminorgrid','off');
text(125,97,textbody,'fontsize',20,'color','r','fontweight','bold');


