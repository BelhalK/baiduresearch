function model=updateThetaFI(model,it)

C=model.param.C;
P=model.param.P;
suffStat=model.suffStat;
theta=model.theta;
Kg=model.param.Kg;

theta(it+1).weight(1)=suffStat.s0(1);
theta(it+1).gamma_2(1)=trace(suffStat.s3(:,:,1))/(2*Kg*suffStat.s0(1));
theta(it+1).alpha(:,1)=pinv(suffStat.s2(:,:,1))*(suffStat.s1(:,1));
theta(it+1).sigma_2(1)=(suffStat.s4(1)-...
    2*theta(it+1).alpha(:,1)'*suffStat.s1(:,1)+...
    theta(it+1).alpha(:,1)'*suffStat.s2(:,:,1)*theta(it+1).alpha(:,1))...
    /(suffStat.s0(1)*P*P);

model.theta(it+1)=theta(it+1);
cd ./tmp_new/
save tmp_fi
cd ..