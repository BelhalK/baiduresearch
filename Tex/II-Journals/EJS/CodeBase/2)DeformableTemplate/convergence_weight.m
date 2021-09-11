function convergence_weight(model,nb_it)


C=model.param.C;
TMP=zeros(C,nb_it);
for n=1:nb_it
    for j=1:C
        TMP(j,n)=model.theta(n).weight(j);
    end
end

colour=hsv(C);
figure,hold on
for j=1:C
    plot([1:nb_it],TMP(j,:),'color',colour(j,:));
end
eval('legend SHOW');