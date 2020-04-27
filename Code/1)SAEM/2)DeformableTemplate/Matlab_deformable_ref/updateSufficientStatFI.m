function model=updateSufficientStatFI(model,model_old,ind)

C=model.param.C;
length = model.length;
coeff_Sto=model.param.STOCHASTIC_STEP_PARAM;
suffStat=model.suffStat;
Kp=model.param.Kp;
Kg=model.param.Kg;
model.countSS=model.countSS+1;

step=1/((model.countSS)^coeff_Sto);

stepfi = 1/(length^(2/3));

countExp=model.countExp;
S0_tmp=model.S0_tmp;
S1_tmp=model.S1_tmp;
S2_tmp=model.S2_tmp;
S3_tmp=model.S3_tmp;
S4_tmp=model.S4_tmp;


S0_tmp_old=model_old.S0_tmp;
S1_tmp_old=model_old.S1_tmp;
S2_tmp_old=model_old.S2_tmp;
S3_tmp_old=model_old.S3_tmp;
S4_tmp_old=model_old.S4_tmp;

Vs0 = model.h.S0;
Vs1 = model.h.S1;
Vs2 = model.h.S2;
Vs3 = model.h.S3;
Vs4 = model.h.S4;

Vs0(1,C,ind) = model.h.S0(1,C,ind) + (S0_tmp(1,C,ind) - S0_tmp_old(1,C,ind))*length/countExp;
Vs1(:,:,ind) = model.h.S1(:,:,ind) + (S1_tmp(:,:,ind) - S1_tmp_old(:,:,ind))*length/countExp;
Vs2(:,:,ind) = model.h.S2(:,:,ind) + (S2_tmp(:,:,ind) - S2_tmp_old(:,:,ind))*length/countExp;
Vs3(:,:,ind) = model.h.S3(:,:,ind) + (S3_tmp(:,:,ind) - S3_tmp_old(:,:,ind))*length/countExp;
Vs4(1,C,ind) = model.h.S4(1,C,ind) + (S4_tmp(1,C,ind) - S4_tmp_old(1,C,ind))*length/countExp;

Vs0_sum = 0;
Vs1_sum = 0;
Vs2_sum = 0;
Vs3_sum = 0;
Vs4_sum = 0;

for index=1:model.length
	Vs0_sum = Vs0_sum + Vs0(1,C,index);
	Vs1_sum = Vs1_sum + Vs1(:,:,index);
	Vs2_sum = Vs2_sum + Vs2(:,:,index);
	Vs3_sum = Vs3_sum + Vs3(:,:,index);
	Vs4_sum = Vs4_sum + Vs4(1,C,index);
end


%update FI statitstics
model.fi.s0(C) = model.fi.s0(C)*(1-stepfi) + stepfi*Vs0_sum/(countExp*length);
model.fi.s1(:,C) = model.fi.s1(:,C)*(1-stepfi) + stepfi*Vs1_sum/(countExp*length);
model.fi.s2(:,:,C) = model.fi.s2(:,:,C)*(1-stepfi) + stepfi*Vs2_sum/(countExp*length);
model.fi.s3(:,:,C) = model.fi.s3(:,:,C)*(1-stepfi) + stepfi*Vs3_sum/(countExp*length);
model.fi.s4(C) = model.fi.s4(C)*(1-stepfi) + stepfi*Vs4_sum/(countExp*length);


%update sufficient statitstics
model.suffStat.s0(C)=suffStat.s0(C)+step*(model.fi.s0(C)-suffStat.s0(C));
model.suffStat.s1(:,C)=suffStat.s1(:,C)+step*(model.fi.s1(:,C)-suffStat.s1(:,C));
model.suffStat.s2(:,:,C)=suffStat.s2(:,:,C)+step*(model.fi.s2(:,:,C)-suffStat.s2(:,:,C));
model.suffStat.s3(:,:,C)=suffStat.s3(:,:,C)+step*(model.fi.s3(:,:,C)-suffStat.s3(:,:,C));
model.suffStat.s4(C)=suffStat.s4(C)+step*(model.fi.s4(C)-suffStat.s4(C));

model.countExp=0;
