dataset='cifar';
model='resnet18';

iid=0;

if iid==1
    ii='_iid1';
else
    ii='_iid0';
end

ll_lamb='_lambda0.0';

num_users_list=[50];
local_ep_list=[1];

sgd_lr='0.01';         % Tinyimagenet
base_lr='1e-05';
lamb_lr='1e-05';
mimelamb_lr='3e-05';

% sgd_lr='0.005';           % cifar
% base_lr='0.0001';
% lamb_lr='0.001';

color_list={'r','b','g','m','c','y'};
dim=3;   
for n=1:length(num_users_list)
    num_users=num_users_list(n);
    for r=1:length(local_ep_list)
        local_ep=local_ep_list(r);
        
        SGD_base_file=dlmread(['./fedlambchecks/checkpoints_' dataset '/' model '_optfedsgd_LAMBFalse_lambda0.0_workers2_lr0.005_epoch100.txt'],'',1,0);
        SGD_base=SGD_base_file(:,dim);
        
        base_file=dlmread(['./fedlambchecks/checkpoints_' dataset '/' model '_optfedams_LAMBFalse_lambda0.0_workers2_lr0.0001_epoch100.txt'],'',1,0);
        base=base_file(:,dim);        
        
        lamb_file=dlmread(['./fedlambchecks/checkpoints_' dataset '/' model '_optfedlamb_LAMBTrue_lambda0.0_workers2_lr0.001_epoch100.txt'],'',1,0);        
        lamb=lamb_file(:,dim);
        T=length(lamb);
        
        reddi_file=dlmread(['./fedlambchecks/checkpoints_' dataset '/resnet18_optreddi_LAMBFalse_lambda0.0_workers2_lr0.005_globlr0.001_epoch100.txt'],'',1,0);              
        reddi=reddi_file(:,dim);

        mime_file=dlmread(['./fedlambchecks/checkpoints_' dataset '/resnet18_optmime_LAMBFalse_lambda0.0_workers2_lr0.0001_epoch100.txt'],'',1,0);              
        mime=mime_file(:,dim);

        mimelamb_file=dlmread(['./fedlambchecks/checkpoints_' dataset '/resnet18_optmime_LAMBTrue_lambda0.0_workers2_lr0.001_epoch100.txt'],'',1,0);              
        mimelamb=mimelamb_file(:,dim);

        
%         lamb(30:end)=lamb(30:end).*(1:(0.02/(length(lamb(30:end))-1)):1.02)';
        figure;
        plot(1:T,base,'k:','linewidth',2);hold on; grid on;
        plot(1:T,SGD_base,'b-.','linewidth',2);hold on; grid on;
        plot(1:T,lamb,'r-','linewidth',2);hold on; grid on;
        plot(1:T,reddi,'g-','linewidth',2);hold on; grid on;

        plot(1:T,mime,'c-','linewidth',2);hold on; grid on;
        plot(1:T,mimelamb,'y-','linewidth',2);hold on; grid on;
        
        legend('Fed-AMS','Fed-SGD','Fed-LAMB','Adp-Fed','MIME','MIME-LAMB','Location','southeast');
        if iid==1
            cc='iid';
        else
            cc='non-iid';
        end
        text(22,lamb(10),['CIFAR + RESNET18' ],'fontsize',20,'fontweight','bold','color','k');
        text(22,lamb(5),['n = 50 EP = ' num2str(1)],'fontsize',20,'fontweight','bold','color','k');
        set(gca,'fontsize',20);
        xlabel('Communication Rounds')
        ylabel('Test Accuracy')
        xticks(0:20:100)
    end
end
