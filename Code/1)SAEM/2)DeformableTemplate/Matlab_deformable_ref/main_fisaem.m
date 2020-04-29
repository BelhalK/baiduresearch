function main_fisaem(data)

model.param=initParam();
model.grid=initGrid(model.param);
model.suffStat=initSuffStat(model.param);
model.fi=initSuffStatFI(model.param);
C=model.param.C;
Kp=model.param.Kp;
Kg=model.param.Kg;
nbItEM=model.param.nbItEM;
phi_mat=phi(model);
sig_init=0.5;
gam_init=0.5;
model.theta(1).alpha(:,1)=pinv(phi_mat'*phi_mat)*(phi_mat'*(mean(data(1:10,:))'));

model.theta(1).sigma_2(1)=sig_init;
model.theta(1).gamma_2(1)=gam_init;
model.theta(1).weight(1)=1/C;
data_start(1,:)=mean(data(1:3,:));
model.length = size(data,1);

tic
model.length = size(data,1);
model.countExp=0;
model.countSS=0;
%init vector of n suff stat
model.S0_tmp=zeros(1,C,model.length);
model.S1_tmp=zeros(Kp,C,model.length);
model.S2_tmp=zeros(Kp,Kp,model.length);
model.S3_tmp=zeros(2*Kg,2*Kg,model.length);
model.S4_tmp=zeros(1,C,model.length);


%init vector of n h stat
model.h.S0=zeros(1,C,model.length);
model.h.S1=zeros(Kp,C,model.length);
model.h.S2=zeros(Kp,Kp,model.length);
model.h.S3=zeros(2*Kg,2*Kg,model.length);
model.h.S4=zeros(1,C,model.length);


nbItEM
for n=1:nbItEM
n
    if n<=model.length
        obs=data(n,:);
        model=simulMissingData_startInc(model,obs,n,n);
        model=updateSufficientStatInc(model);
        model_old = model;
    else
        indices  = randperm(model.length);
        index = indices(1);
        index_j = indices(2);
        obs=data(index,:);
        model=simulMissingDataFI(model,obs,index,n);
        model_old=simulMissingDataFI(model_old,obs,index,index);
        model=updateSufficientStatFI(model, model_old, index);
            
        model=simulMissingDataFI(model,data(index_j,:),index_j,n);
        model_old=simulMissingDataFI(model_old,data(index_j,:),index_j,index_j);
        model.h.S0(1,C,index_j) = model.h.S0(1,C,index_j) + (model.S0_tmp(1,C,index_j) - model_old.S0_tmp(1,C,index_j))/(model.length*model.countExp);
        model.h.S1(:,:,index_j) = model.h.S1(:,:,index_j) + (model.S1_tmp(:,:,index_j) - model_old.S1_tmp(:,:,index_j))/(model.length*model.countExp);
        model.h.S2(:,:,index_j) = model.h.S2(:,:,index_j) + (model.S2_tmp(:,:,index_j) - model_old.S2_tmp(:,:,index_j))/(model.length*model.countExp);
        model.h.S3(:,:,index_j) = model.h.S3(:,:,index_j) + (model.S3_tmp(:,:,index_j) - model_old.S3_tmp(:,:,index_j))/(model.length*model.countExp);
        model.h.S4(1,C,index_j) = model.h.S4(1,C,index_j) + (model.S4_tmp(1,C,index_j) - model_old.S4_tmp(1,C,index_j))/(model.length*model.countExp);
        model_old.theta(index_j) = model.theta(n);
    end
    
    model=updateThetaFI(model,n);
    model.time(n)=toc;
   
end
fprintf("finished\n")

