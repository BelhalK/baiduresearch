function [model carlinChib f_alpha_ref]=simulCarlinChib(model,carlinChib,data,l,i,f_alpha_ref)

C=model.param.C;
P=model.param.P;
Kg=model.param.Kg;
refresh_local=model.param.PROBA_REFRESH_LOCAL;
local=carlinChib.local;
refresh_rigid=model.param.PROBA_REFRESH_RIGID;
rigid=carlinChib.rigid;
refresh_trans=model.param.PROBA_REFRESH_TRANS;
trans=carlinChib.trans;
refresh_homot=model.param.PROBA_REFRESH_HOMOT;
homot=carlinChib.homot;

for j=1:C
    if rand(1)<refresh_local
    [model local(:,j) f_alpha_ref]=simulLocal_MH(model,local(:,j),rigid(:,j),trans(:,j),homot(:,j),data,j,i,f_alpha_ref);
    end
    if rand(1)<refresh_rigid
    [model rigid(:,j) f_alpha_ref]=simulRigid_MH(model,local(:,j),rigid(:,j),trans(:,j),homot(:,j),data,j,i,f_alpha_ref);
    end
    if rand(1)<refresh_trans
    [model trans(:,j) f_alpha_ref]=simulTrans_MH(model,local(:,j),rigid(:,j),trans(:,j),homot(:,j),data,j,i,f_alpha_ref);
    end
    if rand(1)<refresh_homot
    [model homot(:,j) f_alpha_ref]=simulHomot_MH(model,local(:,j),rigid(:,j),trans(:,j),homot(:,j),data,j,i,f_alpha_ref);
    end
    sig2=model.theta(i).sigma_2(j);
    al=model.theta(i).alpha(:,j);
    wei=model.theta(i).weight(j);
    gam=model.theta(i).gamma_2(j);
    var_rigid=model.param.var_rigid;
    var_trans=model.param.var_trans;
    
    poids_est(j)=log(wei)-...
        (P*P/2)*log(sig2)-...
        (1/(2*sig2))*norm(data'-f_alpha_ref(:,j))^2-...
        Kg*log(gam)-...
        (1/(2*gam))*(norm(local(:,j))^2)-...
        (1/(2*var_rigid))*(norm(rigid(:,j))^2)-...
        (1/(2*var_trans))*(norm(trans(:,j))^2);
    
end

M=diag(poids_est)*ones(C)-(diag(poids_est)*ones(C))';
poids_est=1./sum(exp(M));

carlinChib.class=sum(rand(1)>cumsum(poids_est))+1;
carlinChib.local=local;
carlinChib.rigid=rigid;
carlinChib.trans=trans;
carlinChib.homot=homot;