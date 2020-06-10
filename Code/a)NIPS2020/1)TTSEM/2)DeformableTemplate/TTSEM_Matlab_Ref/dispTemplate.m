function dispTemplate(model,it,flag_fig)
C=model.param.C;
phi_mat=phi(model);
if flag_fig
    figure,hold on
end
for j=1:C
    subplot(1,C,j),im(phi_mat*model.theta(it).alpha(:,j),0);
end