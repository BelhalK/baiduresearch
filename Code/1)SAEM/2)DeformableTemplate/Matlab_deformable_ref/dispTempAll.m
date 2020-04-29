function dispTempAll(models,it,flag_fig)
C= length(it);

if flag_fig
    figure,hold on
end


nbmodels = size(models,2);
subplot(nbmodels,C,nbmodels*C)

count = 1;
for j=1:C
	subplot(nbmodels,C,count),im(phi(models(1).model)*models(1).model.theta(j).alpha(:),0);
	count = count+1;
end

for mod=2:nbmodels
	phi_mat=phi(models(mod).model);
	for j=it
		subplot(nbmodels,C,count),im(phi_mat*models(mod).model.theta(j).alpha(:),0);
		count = count+1;
	end
end

