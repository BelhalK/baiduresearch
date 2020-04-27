function model=updateSufficientStatVR(model,iter,ind)

C=model.param.C;
length = model.length;
coeff_Sto=model.param.STOCHASTIC_STEP_PARAM;
suffStat=model.suffStat;
Kp=model.param.Kp;
Kg=model.param.Kg;
model.countSS=model.countSS+1;

step=1/((model.countSS)^coeff_Sto);

stepvr = 1/(length^(2/3));

countExp=model.countExp;
S0_tmp=model.S0_tmp;
S1_tmp=model.S1_tmp;
S2_tmp=model.S2_tmp;
S3_tmp=model.S3_tmp;
S4_tmp=model.S4_tmp;


S0_tmp_e=model.S0_tmp_e;
S1_tmp_e=model.S1_tmp_e;
S2_tmp_e=model.S2_tmp_e;
S3_tmp_e=model.S3_tmp_e;
S4_tmp_e=model.S4_tmp_e;


if mod(iter,length)==0
	model.S0_tmp_e = S0_tmp;
	model.S1_tmp_e = S1_tmp;
	model.S2_tmp_e = S2_tmp;
	model.S3_tmp_e = S3_tmp;
	model.S4_tmp_e = S4_tmp;

	model.S0_tmp_e_sum=0;
	model.S1_tmp_e_sum=0;
	model.S2_tmp_e_sum=0;
	model.S3_tmp_e_sum=0;
	model.S4_tmp_e_sum=0;

	for index=1:length
		model.S0_tmp_e_sum = model.S0_tmp_e_sum + model.S0_tmp_e(1,C,index);
		model.S1_tmp_e_sum = model.S1_tmp_e_sum + model.S1_tmp_e(:,:,index);
		model.S2_tmp_e_sum = model.S2_tmp_e_sum + model.S2_tmp_e(:,:,index);
		model.S3_tmp_e_sum = model.S3_tmp_e_sum + model.S3_tmp_e(:,:,index);
		model.S4_tmp_e_sum = model.S4_tmp_e_sum + model.S4_tmp_e(1,C,index);
	end
end



%update VR statitstics
model.vr.s0(C) = model.vr.s0(C)*(1-stepvr) + stepvr*((S0_tmp(1,C,ind) - S0_tmp_e(1,C,ind))*length/countExp + model.S0_tmp_e_sum/(countExp*length));
model.vr.s1(:,C) = model.vr.s1(:,C)*(1-stepvr) + stepvr*((S1_tmp(:,:,ind) - S1_tmp_e(:,:,ind))*length/countExp + model.S1_tmp_e_sum/(countExp*length));
model.vr.s2(:,:,C) = model.vr.s2(:,:,C)*(1-stepvr) + stepvr*((S2_tmp(:,:,ind) - S2_tmp_e(:,:,ind))*length/countExp + model.S2_tmp_e_sum/(countExp*length));
model.vr.s3(:,:,C) = model.vr.s3(:,:,C)*(1-stepvr) + stepvr*((S3_tmp(:,:,ind) - S3_tmp_e(:,:,ind))*length/countExp + model.S3_tmp_e_sum/(countExp*length));
model.vr.s4(C) = model.vr.s4(C)*(1-stepvr) + stepvr*((S4_tmp(1,C,ind) - S4_tmp_e(1,C,ind))*length/countExp + model.S4_tmp_e_sum/(countExp*length));


%update sufficient statitstics
model.suffStat.s0(C)=suffStat.s0(C)+step*(model.vr.s0(C)-suffStat.s0(C));
model.suffStat.s1(:,C)=suffStat.s1(:,C)+step*(model.vr.s1(:,C)-suffStat.s1(:,C));
model.suffStat.s2(:,:,C)=suffStat.s2(:,:,C)+step*(model.vr.s2(:,:,C)-suffStat.s2(:,:,C));
model.suffStat.s3(:,:,C)=suffStat.s3(:,:,C)+step*(model.vr.s3(:,:,C)-suffStat.s3(:,:,C));
model.suffStat.s4(C)=suffStat.s4(C)+step*(model.vr.s4(C)-suffStat.s4(C));

model.countExp=0;
