function model=simulMissingData(model,data,i)

C=model.param.C;
Kg=model.param.Kg;
nbItMCMC=model.param.nbItMCMC(i);
burnMCMC=model.param.burnMCMC(i);
carlinChib(1).class=randi(C);
carlinChib(1).local=zeros(2*Kg,C);
carlinChib(1).rigid=zeros(3,C);
carlinChib(1).trans=zeros(2,C);
carlinChib(1).homot=[ones(1,C);zeros(2,C)];

f_alpha_ref(:)=deformatedTemplateKernel(carlinChib(1).local(:),...
        carlinChib(1).rigid(:),...
        carlinChib(1).trans(:),...
        carlinChib(1).homot(:),...
        model)*model.theta(i).alpha(:);


for l=2:nbItMCMC
    [model carlinChib(l) f_alpha_ref]=simulCarlinChib(model,carlinChib(l-1),data,l,i,f_alpha_ref); 
end

for l=burnMCMC+1:nbItMCMC
    class=carlinChib(l).class;
    local=carlinChib(l).local(:,class);
    rigid=carlinChib(l).rigid(:,class);
    trans=carlinChib(l).trans(:,class);
    homot=carlinChib(l).homot(:,class);
    PHI_DEF=deformatedTemplateKernel(local,rigid,trans,homot,model);
    model.S0_tmp(class)=model.S0_tmp(class)+1;
    
    model.S1_tmp(:)=model.S1_tmp(:)+PHI_DEF'*data';
    model.S2_tmp(:,:)=model.S2_tmp(:,:)+PHI_DEF'*PHI_DEF;
    model.S3_tmp(:,:)=model.S3_tmp(:,:)+local*local';
    model.S4_tmp(class)=model.S4_tmp(class)+norm(data)^2;
    model.countExp=model.countExp+1;
end
