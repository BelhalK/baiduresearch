function [model homot f_alpha_ref]=simulHomot_MH(model,local,rigid,trans,homot,obs,class,n,f_alpha_ref)

SIG_RANDOM_WALK_HOMOT=diag(model.param.SIG_RANDOM_WALK_HOMOT);

proposal=SIG_RANDOM_WALK_HOMOT*randn(3,1);

proposal=homot+proposal;

[log_acceptation f_alpha_proposal]=log_acceptance_homot(model,local,rigid,trans,homot,proposal,obs',class,n,f_alpha_ref);

log_U=log(rand(1,1));
    
    if (log_U<log_acceptation)
        homot=proposal;
        f_alpha_ref(:,class)=f_alpha_proposal;
        model.param.acceptationMH_homot=model.param.acceptationMH_homot+1;
    end
    
model.param.totalMH_homot=model.param.totalMH_homot+1;