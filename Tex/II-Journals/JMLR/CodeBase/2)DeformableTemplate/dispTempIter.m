function dispTempIter(model,it,flag_fig)
C= length(it);
phi_mat=phi(model);
if flag_fig
    figure,hold on
end

count = 1;
for j=it
	subplot(1,C,count),im(phi_mat*model.theta(j).alpha(:),0);
	count = count+1;
end
