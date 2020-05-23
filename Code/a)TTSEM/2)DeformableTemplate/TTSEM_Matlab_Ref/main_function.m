function main_function(data)

model.param=initParam();
model.grid=initGrid(model.param);
model.suffStat=initSuffStat(model.param);
C=model.param.C;
Kp=model.param.Kp;
Kg=model.param.Kg;
nbItEM=model.param.nbItEM;
phi_mat=phi(model);
sig_init=0.5;
gam_init=0.5;
model.theta(1).alpha(:,1)=pinv(phi_mat'*phi_mat)*(phi_mat'*(mean(data(1:5,:))'));
model.theta(1).alpha(:,2)=pinv(phi_mat'*phi_mat)*(phi_mat'*(mean(data(21:23,:))'));
model.theta(1).alpha(:,3)=pinv(phi_mat'*phi_mat)*(phi_mat'*(mean(data(43:48,:))'));
model.theta(1).alpha(:,4)=pinv(phi_mat'*phi_mat)*(phi_mat'*(mean(data(74:77,:))'));
model.theta(1).alpha(:,5)=pinv(phi_mat'*phi_mat)*(phi_mat'*(mean(data(87:90,:))'));
model.theta(1).alpha(:,6)=pinv(phi_mat'*phi_mat)*(phi_mat'*(mean(data(55:60,:))'));
model.theta(1).alpha(:,7)=pinv(phi_mat'*phi_mat)*(phi_mat'*(mean(data(95:100,:))'));

for j=1:C
    model.theta(1).sigma_2(j)=sig_init;
    model.theta(1).gamma_2(j)=gam_init;
    model.theta(1).weight(j)=1/C;
end

data_start(1,:)=mean(data(1:3,:));
data_start(2,:)=mean(data(21:23,:));
data_start(3,:)=mean(data(43:45,:));
data_start(4,:)=mean(data(61:63,:));
data_start(5,:)=mean(data(87:90,:));
data_start(6,:)=mean(data(55:57,:));
data_start(7,:)=mean(data(95:97,:));

tic
model.countExp=0;
model.countSS=0;
model.S0_tmp=zeros(1,C);
model.S1_tmp=zeros(Kp,C);
model.S2_tmp=zeros(Kp,Kp,C);
model.S3_tmp=zeros(2*Kg,2*Kg,C);
model.S4_tmp=zeros(1,C);

for n=1:nbItEM
n
    if n<=C
        obs=data_start(n,:);
        model=simulMissingData_start(model,obs,n);
    else
        obs=data(randi(nbItEM),:);
        model=simulMissingData(model,obs,n);
    end
    if sum(n==model.param.updateSuffStat)
        model=updateSufficientStat(model);
    end
    if sum(n==model.param.updateTheta)
        model=updateTheta(model,n);
        model.time(n)=toc;
    else
        model.theta(n+1)=model.theta(n);
    end
   
end