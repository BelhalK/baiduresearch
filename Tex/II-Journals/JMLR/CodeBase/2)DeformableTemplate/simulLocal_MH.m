function [model local f_alpha_ref]=simulLocal_MH(model,local,rigid,trans,homot,obs,class,it,f_alpha_ref)

Kg=model.param.Kg;
SIG_RANDOM_WALK_LOCAL=model.param.SIG_RANDOM_WALK_LOCAL;


v=randi(2*Kg);   
move=SIG_RANDOM_WALK_LOCAL*randn(2*Kg,1);
proposal=local+move.*([1:2*Kg]==v)';
[log_acceptation f_alpha_proposal]=log_acceptance_local(model,local,rigid,trans,homot,proposal,obs',class,it,f_alpha_ref);
log_U=log(rand(1));

if (log_U<log_acceptation)
    local=proposal;
    f_alpha_ref(:,class)=f_alpha_proposal;
    model.param.acceptationMH_local=model.param.acceptationMH_local+1;
end

model.param.totalMH_local=model.param.totalMH_local+1;
