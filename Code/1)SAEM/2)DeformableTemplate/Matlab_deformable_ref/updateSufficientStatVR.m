function model=updateSufficientStatVR(model)

C=model.param.C;
length = model.length;
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

S0_tmp_sum=0;
S1_tmp_sum=0;
S2_tmp_sum=0;
S3_tmp_sum=0;
S4_tmp_sum=0;

for index=1:length
	S0_tmp_sum = S0_tmp_sum + S0_tmp(1,C,index);
	S1_tmp_sum = S1_tmp_sum + S1_tmp(:,:,index);
	S2_tmp_sum = S2_tmp_sum + S2_tmp(:,:,index);
	S3_tmp_sum = S3_tmp_sum + S3_tmp(:,:,index);
	S4_tmp_sum = S4_tmp_sum + S4_tmp(1,C,index);
end


model.suffStat.s0(C)=suffStat.s0(C)+step*(S0_tmp_sum/(countExp*model.length)-suffStat.s0(C));
model.suffStat.s1(:,C)=suffStat.s1(:,C)+step*(S1_tmp_sum/(countExp*model.length)-suffStat.s1(:,C));
model.suffStat.s2(:,:,C)=suffStat.s2(:,:,C)+step*(S2_tmp_sum/(countExp*model.length)-suffStat.s2(:,:,C));
model.suffStat.s3(:,:,C)=suffStat.s3(:,:,C)+step*(S3_tmp_sum/(countExp*model.length)-suffStat.s3(:,:,C));
model.suffStat.s4(C)=suffStat.s4(C)+step*(S4_tmp_sum/(countExp*model.length)-suffStat.s4(C));

model.countExp=0;
