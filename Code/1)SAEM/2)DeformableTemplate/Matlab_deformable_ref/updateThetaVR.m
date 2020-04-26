function model=updateThetaVR(model,it)

C=model.param.C;
P=model.param.P;
suffStat=model.suffStat;
theta=model.theta;
Kg=model.param.Kg;

for j=1:C
    theta(it+1).weight(j)=suffStat.s0(j);
    theta(it+1).gamma_2(j)=trace(suffStat.s3(:,:,j))/(2*Kg*suffStat.s0(j));
    theta(it+1).alpha(:,j)=pinv(suffStat.s2(:,:,j))*(suffStat.s1(:,j));
    theta(it+1).sigma_2(j)=(suffStat.s4(j)-...
        2*theta(it+1).alpha(:,j)'*suffStat.s1(:,j)+...
        theta(it+1).alpha(:,j)'*suffStat.s2(:,:,j)*theta(it+1).alpha(:,j))...
        /(suffStat.s0(j)*P*P);
end

model.theta(it+1)=theta(it+1);
cd ./tmp_new/
save tmp_vr
cd ..