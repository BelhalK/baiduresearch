function convergence_gamma(model,nb_it,flag_fig)
if flag_fig
figure,hold on;
end
C=model.param.C;
col=hsv(C);
for n=1:nb_it
    for j=1:C
    gam(n,j)=sqrt(model.theta(n).gamma_2(j));
    end    
end

for j=1:C
    plot([1:nb_it],gam(:,j)','color',col(j,:));
end
eval('legend SHOW');