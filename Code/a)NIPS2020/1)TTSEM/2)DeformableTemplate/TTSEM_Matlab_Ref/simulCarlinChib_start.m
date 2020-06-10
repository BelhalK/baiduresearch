function [model carlinChib f_alpha_ref]=simulCarlinChib_start(model,carlinChib,data,l,i,f_alpha_ref)

C=model.param.C;
local=carlinChib.local;
rigid=carlinChib.rigid;
trans=carlinChib.trans;
homot=carlinChib.homot;
for j=(mod(i,C))*(mod(i,C)>0)+C*(mod(i,C)==0);
    [model local(:,j) f_alpha_ref]=simulLocal_MH(model,local(:,j),rigid(:,j),trans(:,j),homot(:,j),data,j,i,f_alpha_ref);
    [model rigid(:,j) f_alpha_ref]=simulRigid_MH(model,local(:,j),rigid(:,j),trans(:,j),homot(:,j),data,j,i,f_alpha_ref);
    [model trans(:,j) f_alpha_ref]=simulTrans_MH(model,local(:,j),rigid(:,j),trans(:,j),homot(:,j),data,j,i,f_alpha_ref);
    [model homot(:,j) f_alpha_ref]=simulHomot_MH(model,local(:,j),rigid(:,j),trans(:,j),homot(:,j),data,j,i,f_alpha_ref);
end
   
carlinChib.class=(mod(i,C))*(mod(i,C)>0)+C*(mod(i,C)==0);
carlinChib.local=local;
carlinChib.rigid=rigid;
carlinChib.trans=trans;
carlinChib.homot=homot;