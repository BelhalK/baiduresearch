function model=updateSufficientStatInc(model)

C=model.param.C;
coeff_Sto=model.param.STOCHASTIC_STEP_PARAM;
suffStat=model.suffStat;
Kp=model.param.Kp;
Kg=model.param.Kg;
model.countSS=model.countSS+1;

step=1/((model.countSS)^coeff_Sto);

countExp=model.countExp;
S0_tmp= model.S0_tmp;
S1_tmp=model.S1_tmp;
S2_tmp=model.S2_tmp;
S3_tmp=model.S3_tmp;
S4_tmp=model.S4_tmp;

for j=1:C
    model.suffStat.s0(j)=suffStat.s0(j)+step*(S0_tmp(j)/countExp-suffStat.s0(j));
    model.suffStat.s1(:,j)=suffStat.s1(:,j)+step*(S1_tmp(:,j)/countExp-suffStat.s1(:,j));
    model.suffStat.s2(:,:,j)=suffStat.s2(:,:,j)+step*(S2_tmp(:,:,j)/countExp-suffStat.s2(:,:,j));
    model.suffStat.s3(:,:,j)=suffStat.s3(:,:,j)+step*(S3_tmp(:,:,j)/countExp-suffStat.s3(:,:,j));
    model.suffStat.s4(j)=suffStat.s4(j)+step*(S4_tmp(j)/countExp-suffStat.s4(j));
end

model.countExp=0;


oldS0 = model.S0_tmp

model.S0_tmp=zeros(1,C);
model.S1_tmp=zeros(Kp,C);
model.S2_tmp=zeros(Kp,Kp,C);
model.S3_tmp=zeros(2*Kg,2*Kg,C);
model.S4_tmp=zeros(1,C);
