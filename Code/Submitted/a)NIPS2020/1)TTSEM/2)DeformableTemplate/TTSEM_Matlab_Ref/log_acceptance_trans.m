function [log_acc f_alpha_proposal]=log_acceptance_trans(model,local,rigid,trans,homot,proposal,obs,class,it,f_alpha_ref)

alpha=model.theta(it).alpha(:,class);
sigma_2=model.theta(it).sigma_2(class);
var_trans=model.param.var_trans;

f_alpha_proposal=deformatedTemplateKernel(local,rigid,proposal,homot,model)*alpha;

log_acc=-(1/(2*sigma_2))*norm(obs-f_alpha_proposal)^2-...
    (1/(2*var_trans))*norm(proposal)^2+...
    (1/(2*sigma_2))*norm(obs-f_alpha_ref(:,class))^2+...
    (1/(2*var_trans))*norm(trans)^2;