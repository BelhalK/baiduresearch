function main_online(data)

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
model.theta(1).alpha(:,1)=pinv(phi_mat'*phi_mat)*(phi_mat'*(mean(data(:,:))'));


model.theta(1).sigma_2(1)=sig_init;
model.theta(1).gamma_2(1)=gam_init;
model.theta(1).weight(1)=1/C;
data_start(1,:)=mean(data(1:3,:));

tic
model.countExp=0;
model.countSS=0;
model.S0_tmp=zeros(1,C);
model.S1_tmp=zeros(Kp,C);
model.S2_tmp=zeros(Kp,Kp);
model.S3_tmp=zeros(2*Kg,2*Kg);
model.S4_tmp=zeros(1,C);

length = size(data,1);

nbItEM
for n=1:nbItEM
n
    if n<=C
        obs=data_start(n,:);
        model=simulMissingData_start(model,obs,n);
    else
        obs=data(randi(length),:);
        model=simulMissingData(model,obs,n);
    end
    model=updateSufficientStat(model);
    model=updateTheta(model,n);
    model.time(n)=toc;
    %model.theta(n+1)=model.theta(n);
   
end
fprintf("finished\n")