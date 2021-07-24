function [log_acc f_alpha_proposal]=log_acceptance_local(model,local,rigid,trans,homot,proposal,obs,class,it,f_alpha_ref)

alpha=model.theta(it).alpha(:,class);
sigma_2=model.theta(it).sigma_2(class);
gamma_2=model.theta(it).gamma_2(class);

f_alpha_proposal=deformatedTemplateKernel(proposal,rigid,trans,homot,model)*alpha;

log_acc=-(1/(2*sigma_2))*norm(obs-f_alpha_proposal)^2-...
    (1/(2*gamma_2))*norm(proposal)^2+...
    (1/(2*sigma_2))*norm(obs-f_alpha_ref(:,class))^2+...
    (1/(2*gamma_2))*norm(local)^2;