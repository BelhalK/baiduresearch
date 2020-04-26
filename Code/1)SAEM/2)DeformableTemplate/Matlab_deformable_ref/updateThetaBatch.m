function model=updateThetaBatch(model,it)

C=model.param.C;
P=model.param.P;
suffStat=model.suffStat;
theta=model.theta;
Kg=model.param.Kg;

theta(it+1).weight(1)=suffStat.s0(1);
theta(it+1).gamma_2(1)=trace(suffStat.s3(:,:))/(2*Kg*suffStat.s0(1));
theta(it+1).alpha(:)=pinv(suffStat.s2(:,:))*(suffStat.s1(:));
theta(it+1).sigma_2(1)=(suffStat.s4(1)-...
	2*theta(it+1).alpha(:)'*suffStat.s1(:)+...
	theta(it+1).alpha(:)'*suffStat.s2(:,:)*theta(it+1).alpha(:))...
	/(suffStat.s0(1)*P*P);

model.theta(it+1)=theta(it+1);
cd ./tmp_new/
save tmp_batch
cd ..