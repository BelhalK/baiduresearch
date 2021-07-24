function [log_acc f_alpha_proposal]=log_acceptance_homot(model,local,rigid,trans,homot,proposal,obs,class,it,f_alpha_ref)

alpha=model.theta(it).alpha(:,class);
sigma_2=model.theta(it).sigma_2(class);

f_alpha_proposal=deformatedTemplateKernel(local,rigid,trans,proposal,model)*alpha;

log_acc=-(1/(2*sigma_2))*norm(obs-f_alpha_proposal)^2+...
    (1/(2*sigma_2))*norm(obs-f_alpha_ref(:,class))^2;