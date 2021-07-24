function [model poids_est]=compute_approx_weight_bis(model,obs,ind_obs,i)

C=model.param.C;
Kg=model.param.Kg;
P=model.param.P;

defo=zeros(2*Kg,C);

for j=1:C
    defo(:,j)=model.defo_aux(ind_obs,:,j)';
end

poids_est=zeros(C,1);

for j=1:C
    sig2=model.theta(i).sigma_2(j);
    al=model.theta(i).alpha(:,j);
    wei=model.theta(i).weight(j);
    gam=model.theta(i).gamma_2(:,:,j);
    
    [model defo(:,j)]=simulDefo_MH(model,defo(:,j),obs,j,i);
    phi_def=deformatedTemplateKernel(defo(:,j),model);
    
    poids_est(j)=log(wei)-...
        (P*P/2)*log(sig2)-...
        (1/(2*sig2))*norm(obs'-phi_def*al)^2-...
        (1/2)*log(det(gam))-...
        (1/2)*(defo(:,j)'*(inv(gam)*defo(:,j)));
    
    model.defo_aux(ind_obs,:,j)=defo(:,j)';
end

M=diag(poids_est)*ones(C)-(diag(poids_est)*ones(C))';
poids_est=1./sum(exp(M));
