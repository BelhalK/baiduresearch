function main_vrsaem(data)

model.param=initParam();
model.grid=initGrid(model.param);
model.suffStat=initSuffStat(model.param);
model.vr=initSuffStatVR(model.param);
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

%init vector of n VR stat
model.S0_tmp_e=zeros(1,C,model.length);
model.S1_tmp_e=zeros(Kp,C,model.length);
model.S2_tmp_e=zeros(Kp,Kp,model.length);
model.S3_tmp_e=zeros(2*Kg,2*Kg,model.length);
model.S4_tmp_e=zeros(1,C,model.length);

model.S0_tmp_e_sum=0;
model.S1_tmp_e_sum=0;
model.S2_tmp_e_sum=0;
model.S3_tmp_e_sum=0;
model.S4_tmp_e_sum=0;

for index=1:model.length
    model.S0_tmp_e_sum = model.S0_tmp_e_sum + model.S0_tmp_e(1,C,index);
    model.S1_tmp_e_sum = model.S1_tmp_e_sum + model.S1_tmp_e(:,:,index);
    model.S2_tmp_e_sum = model.S2_tmp_e_sum + model.S2_tmp_e(:,:,index);
    model.S3_tmp_e_sum = model.S3_tmp_e_sum + model.S3_tmp_e(:,:,index);
    model.S4_tmp_e_sum = model.S4_tmp_e_sum + model.S4_tmp_e(1,C,index);
end



nbItEM
for n=1:nbItEM
n
    if n<=C
        obs=data_start(n,:);
        model=simulMissingData_startInc(model,obs,n,n);
    else
        index = randi(model.length);
        obs=data(index,:);
        model=simulMissingDataVR(model,obs,index,n);
    end
    model=updateSufficientStatVR(model,n, index);
    model=updateThetaVR(model,n);
    model.time(n)=toc;
    %model.theta(n+1)=model.theta(n);
   
end
fprintf("finished\n")

