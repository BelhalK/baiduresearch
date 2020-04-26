function model=simulMissingData_startInc(model,data,index,i)

C=model.param.C;
Kg=model.param.Kg;
nbItMCMC=20;%model.param.nbItMCMC(i);
burnMCMC=0;%model.param.burnMCMC(i);
carlinChib(1).class=(mod(i,C))*(mod(i,C)>0)+C*(mod(i,C)==0);
carlinChib(1).local=zeros(2*Kg,C);
carlinChib(1).rigid=zeros(3,C);
carlinChib(1).trans=zeros(2,C);
carlinChib(1).homot=[ones(1,C);zeros(2,C)];

for j=1:C
    f_alpha_ref(:,j)=deformatedTemplateKernel(carlinChib(1).local(:,j),...
        carlinChib(1).rigid(:,j),carlinChib(1).trans(:,j),...
        carlinChib(1).homot(:,j),model)*model.theta(i).alpha(:,j);
end


for l=2:nbItMCMC
    [model carlinChib(l) f_alpha_ref]=simulCarlinChib_start(model,carlinChib(l-1),data,l,i,f_alpha_ref); 
end

for l=burnMCMC+1:nbItMCMC
    class=carlinChib(l).class;
    local=carlinChib(l).local(:,class);
    rigid=carlinChib(l).rigid(:,class);
    trans=carlinChib(l).trans(:,class);
    homot=carlinChib(l).homot(:,class);
    PHI_DEF=deformatedTemplateKernel(local,rigid,trans,homot,model);
    model.S0_tmp(class,1,index)=model.S0_tmp(class)+1;
    model.S1_tmp(:,class,index)=model.S1_tmp(:,class,index)+PHI_DEF'*data';
    model.S2_tmp(:,:,index)=model.S2_tmp(:,:,index)+PHI_DEF'*PHI_DEF;
    model.S3_tmp(:,:,index)=model.S3_tmp(:,:,index)+local*local';
    model.S4_tmp(class,1,index)=model.S4_tmp(class,1,index)+norm(data)^2;
    model.countExp=model.countExp+1;
end
