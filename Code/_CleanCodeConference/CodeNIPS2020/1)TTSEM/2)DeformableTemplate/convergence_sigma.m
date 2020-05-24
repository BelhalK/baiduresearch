function convergence_sigma(model,nb_it,flag_fig)
if flag_fig
figure,hold on;
end
C=model.param.C;
col=hsv(C);
for n=1:nb_it
    for j=1:C
    sig(n,j)=sqrt(model.theta(n).sigma_2(j));
    end    
end

for j=1:C
    plot([1:nb_it],sig(:,j)','color',col(j,:));
end
eval('legend SHOW');