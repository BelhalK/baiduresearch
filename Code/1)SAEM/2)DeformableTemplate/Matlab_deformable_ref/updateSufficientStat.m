function model=updateSufficientStat(model)

C=model.param.C;
coeff_Sto=model.param.STOCHASTIC_STEP_PARAM;
suffStat=model.suffStat;
Kp=model.param.Kp;
Kg=model.param.Kg;
model.countSS=model.countSS+1;

step=1/((model.countSS)^coeff_Sto);

countExp=model.countExp;
S0_tmp=model.S0_tmp;
S1_tmp=model.S1_tmp;
S2_tmp=model.S2_tmp;
S3_tmp=model.S3_tmp;
S4_tmp=model.S4_tmp;

model.suffStat.s0(1)=suffStat.s0(1)+step*(S0_tmp(1)/countExp-suffStat.s0(1));
model.suffStat.s1(:)=suffStat.s1(:)+step*(S1_tmp(:)/countExp-suffStat.s1(:));
model.suffStat.s2(:,:)=suffStat.s2(:,:)+step*(S2_tmp(:,:)/countExp-suffStat.s2(:,:));
model.suffStat.s3(:,:)=suffStat.s3(:,:)+step*(S3_tmp(:,:)/countExp-suffStat.s3(:,:));
model.suffStat.s4(1)=suffStat.s4(1)+step*(S4_tmp(1)/countExp-suffStat.s4(1));


model.countExp=0;
model.S0_tmp=zeros(1,C);
model.S1_tmp=zeros(Kp,C);
model.S2_tmp=zeros(Kp,Kp);
model.S3_tmp=zeros(2*Kg,2*Kg);
model.S4_tmp=zeros(1,C);