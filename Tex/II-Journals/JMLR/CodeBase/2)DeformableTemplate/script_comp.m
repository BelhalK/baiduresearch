for rr=1:n
    dispTemplate(model,rr,0);
    rr
    pause(0.15)
end
%%
for j=1:2
   def=mvnrnd(zeros(72,1),model.theta(n).gamma_2(:,:,j));
   subplot(1,2,j), im(deformatedTemplateKernel(def',model)*model.theta(n).alpha(:,j),0);
end