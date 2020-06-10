function [model rigid f_alpha_ref]=simulRigid_MH(model,local,rigid,trans,homot,obs,class,n,f_alpha_ref)

SIG_RANDOM_WALK_RIGID=diag(model.param.SIG_RANDOM_WALK_RIGID);

proposal=SIG_RANDOM_WALK_RIGID*randn(3,1);

proposal=rigid+proposal;

[log_acceptation f_alpha_proposal]=log_acceptance_rigid(model,local,rigid,trans,homot,proposal,obs',class,n,f_alpha_ref);

log_U=log(rand(1,1));
    
    if (log_U<log_acceptation)
        rigid=proposal;
        f_alpha_ref(:,class)=f_alpha_proposal;
        model.param.acceptationMH_rigid=model.param.acceptationMH_rigid+1;
    end
    
model.param.totalMH_rigid=model.param.totalMH_rigid+1;