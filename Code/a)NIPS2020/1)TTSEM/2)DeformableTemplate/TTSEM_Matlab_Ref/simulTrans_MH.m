function [model trans f_alpha_ref]=simulTrans_MH(model,local,rigid,trans,homot,obs,class,n,f_alpha_ref)

SIG_RANDOM_WALK_TRANS=diag(model.param.SIG_RANDOM_WALK_TRANS);

proposal=SIG_RANDOM_WALK_TRANS*randn(2,1);

proposal=trans+proposal;

[log_acceptation f_alpha_proposal]=log_acceptance_trans(model,local,rigid,trans,homot,proposal,obs',class,n,f_alpha_ref);

log_U=log(rand(1,1));
    
    if (log_U<log_acceptation)
        trans=proposal;
        f_alpha_ref(:,class)=f_alpha_proposal;
        model.param.acceptationMH_trans=model.param.acceptationMH_trans+1;
    end
    
model.param.totalMH_trans=model.param.totalMH_trans+1;