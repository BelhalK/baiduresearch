function [log_acc f_alpha_proposal]=log_acceptance_rigid(model,local,rigid,trans,homot,proposal,obs,class,it,f_alpha_ref)

alpha=model.theta(it).alpha(:,class);
sigma_2=model.theta(it).sigma_2(class);
var_rigid=model.param.var_rigid;

f_alpha_proposal=deformatedTemplateKernel(local,proposal,trans,homot,model)*alpha;

log_acc=-(1/(2*sigma_2))*norm(obs-f_alpha_proposal)^2-...
    (1/(2*var_rigid))*norm(proposal)^2+...
    (1/(2*sigma_2))*norm(obs-f_alpha_ref(:,class))^2+...
    (1/(2*var_rigid))*norm(rigid)^2;