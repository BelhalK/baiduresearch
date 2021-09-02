n = numel(test_acc_adam_0001);
t = 1:250:250*n;
figure(2)
plot(t(1:10:end),test_acc_adam_01(1:10:end),'-*')
hold on

plot(t(1:10:end),test_acc_adam_001(1:10:end),'-+')
hold on

plot(t(1:10:end),test_acc_adam_0001(1:10:end),'-v')
hold on

plot(t(1:10:end),test_acc_adam_00001(1:10:end),'-o')

plot(t(1:10:end),test_acc_adam_000001(1:10:end),'-x')

plot(t(1:10:end),test_acc_adam_0000001(1:10:end),'-s')
xlabel('number of iterations')
ylabel('test accuracy')
legend('1e-1','1e-2','1e-3','1e-4','1e-5','1e-6')




figure(1)
plot(t(1:10:end),test_loss_adam_01(1:10:end),'-*')
hold on

plot(t(1:10:end),test_loss_adam_001(1:10:end),'-+')
hold on

plot(t(1:10:end),test_loss_adam_0001(1:10:end),'-v')
hold on

plot(t(1:10:end),test_loss_adam_00001(1:10:end),'-o')

plot(t(1:10:end),test_loss_adam_000001(1:10:end),'-x')

plot(t(1:10:end),test_loss_adam_0000001(1:10:end),'-s')





xlabel('number of iterations')
ylabel('test loss')
legend('1e-1','1e-2','1e-3','1e-4','1e-5','1e-6')


