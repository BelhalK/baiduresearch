function plot_cifar
 
folder = './';textbody = 'CelebA Dataset';
 
 
file = importdata([folder 'fid_celeb50k.txt']);
 
x= file.data(:,1)
vanilla = file.data(:,2)
stanley = file.data(:,3)
hmc = file.data(:,4)
gd = file.data(:,5)

 
figure;box on; grid on; 
plot(x,vanilla,'b-','linewidth',1.5);hold on; grid on;  
plot(x,stanley,'r-','linewidth',1.5);hold on; grid on;  
plot(x,hmc,'k-','linewidth',1.5);hold on; grid on;  
plot(x,gd,'g-','linewidth',1.5);hold on; grid on;  

xlim([1000 500000])
set(gca,'fontsize',20);
xlabel('Iterations');
ylabel('FID');
 
legend('Langevin','STANLey','HMC','GD - no noise','location','northeast');
set(gca,'xminorgrid','off');
text(10000,120,textbody,'fontsize',20,'color','m','fontweight','bold');

