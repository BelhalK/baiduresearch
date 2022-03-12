function plot_CIFAR

folder = 'CIFAR/resnet/';textbody = 'CIFAR: Resnet';
%folder = 'CIFAR/vggnet/';textbody = 'CIFAR: VGG';

x = [100, 200, 1000,2000, 5000, 10000, 20000,30000, 50000]

adabound = importdata([folder 'adabound_loss.txt']);
adam = importdata([folder 'adam_loss.txt']);
rmsprop = importdata([folder 'rmsprop_loss.txt']);
sgd = importdata([folder 'sgd_loss.txt']);
sagd = importdata([folder 'sagd_loss.txt']);

figure;box on; grid on; 
semilogx(x,adabound,'b-o','linewidth',1.5);hold on; grid on;
plot(x,adam,'g-+','linewidth',1.5);hold on; grid on;  
plot(x,rmsprop,'k-*','linewidth',1.5);hold on; grid on;  
plot(x,sgd,'m-<','linewidth',1.5);hold on; grid on;  
plot(x,sagd,'r-d','linewidth',1.5);hold on; grid on;  

set(gca,'fontsize',20);
xlabel('# Samples');
ylabel('Testing Loss');
legend('AdaBound','Adam','RMSprop','SGD','SAGD','location','northeast');
set(gca,'xminorgrid','off');
text(135,1.3,textbody,'fontsize',20,'color','r','fontweight','bold');


adabound = importdata([folder 'adabound_acc.txt']);
adam = importdata([folder 'adam_acc.txt']);
rmsprop = importdata([folder 'rmsprop_acc.txt']);
sgd = importdata([folder 'sgd_acc.txt']);
sagd = importdata([folder 'sagd_acc.txt']);



figure;box on; grid on; 
semilogx(x,adabound,'b-o','linewidth',1.5);hold on; grid on;
plot(x,adam,'g-+','linewidth',1.5);hold on; grid on;  
plot(x,rmsprop,'k-*','linewidth',1.5);hold on; grid on;  
plot(x,sgd,'m-<','linewidth',1.5);hold on; grid on;  
plot(x,sagd,'r-d','linewidth',1.5);hold on; grid on;  

set(gca,'fontsize',20);
xlabel('# Samples');
ylabel('Accuracy (%)');
legend('AdaBound','Adam','RMSprop','SGD','SAGD','location','southeast');
set(gca,'xminorgrid','off');
text(125,97,textbody,'fontsize',20,'color','r','fontweight','bold');
