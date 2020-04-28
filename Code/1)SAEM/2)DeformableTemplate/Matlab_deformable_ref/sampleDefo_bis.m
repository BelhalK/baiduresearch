function [model defo]=sampleDefo_bis(model,obs,class,ind_obs,i)
Kg=model.param.Kg;
nbItMCMC=model.param.nbItMCMC;
defo=model.defo_aux(ind_obs,:,class)';

for l=nbItMCMC
    [model defo]=simulDefo_MH(model,defo,obs,class,i);
end


model.defo_aux(ind_obs,:,class)=defo';