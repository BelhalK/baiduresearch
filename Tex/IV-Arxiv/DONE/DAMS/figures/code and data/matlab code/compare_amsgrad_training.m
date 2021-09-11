n = numel(avg_loss_amsgrad_001);
t = 1:5:n-1;
figure(7)
train_loss_01 = avg_every_5(avg_loss_amsgrad_01);
train_loss_001 = avg_every_5(avg_loss_amsgrad_001);
train_loss_0001 = avg_every_5(avg_loss_amsgrad_0001);
train_loss_00001 = avg_every_5(avg_loss_amsgrad_00001);
train_loss_000001 = avg_every_5(avg_loss_amsgrad_000001);
train_loss_0000001 = avg_every_5(avg_loss_amsgrad_0000001);
    


plot(t(1:250:end),train_loss_01(1:250:end),'-*')
hold on 
plot(t(1:250:end),train_loss_001(1:250:end),'-+')


plot(t(1:250:end),train_loss_0001(1:250:end),'-v')
hold on

plot(t(1:250:end),train_loss_00001(1:250:end),'-o')

plot(t(1:250:end),train_loss_000001(1:250:end),'-x')

plot(t(1:250:end),train_loss_0000001(1:250:end),'-s')


ylim([0,2.6])

xlabel('number of iterations')
ylabel('training loss')
legend('1e-1','1e-2','1e-3','1e-4','1e-5','1e-6')


function res = avg_every_5(vector)
n = numel(vector);
length = n/5;
res = zeros(1,length);
for i = 1:length
    start_id = (i-1)*5 + 1;
    res(i) = mean(vector(start_id:start_id+4));
end
end