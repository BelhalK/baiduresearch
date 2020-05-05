function dispDefTemplate(model,obs,carlinChib,it,flag_fig)
C=model.param.C;
P=model.param.P;
var_trans=model.param.var_trans;
var_rigid=model.param.var_rigid;
local=carlinChib.local;
rigid=carlinChib.rigid;
trans=carlinChib.trans;
homot=carlinChib.homot;

if flag_fig
    figure,hold on
end
for j=1:C
    phi_def=deformatedTemplateKernel(local(:,j),rigid(:,j),trans(:,j),homot(:,j),model);
    subplot(1,C+1,j),im(phi_def*model.theta(it).alpha(:,j),0);
    R(1,j)=log(model.theta(it).weight(j));
    R(2,j)=-(1/(2*model.theta(it).sigma_2(j)))*norm(obs-phi_def*model.theta(it).alpha(:,j))^2;
    R(3,j)=-(P*P/2)*log(model.theta(it).sigma_2(j));
    R(4,j)=-(1/2)*(local(:,j)'*(pinv(model.theta(it).gamma_2(j))*local(:,j)));
    R(5,j)=-(1/2)*log(det(model.theta(it).gamma_2(j)));
    R(6,j)=-(1/(2*var_rigid))*norm(rigid(:,j))^2;
    R(7,j)=-(1/(2*var_trans))*norm(trans(:,j))^2;
    R(8,j)=sum(R(:,j),1);
end
subplot(1,C+1,C+1),im(obs,0);
R