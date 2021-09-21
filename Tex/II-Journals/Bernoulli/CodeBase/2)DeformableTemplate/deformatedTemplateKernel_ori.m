function PHI_DEF=deformatedTemplateKernel_ori(local,rigid,trans,model)


SIG_PHO=model.param.SIG_PHO;
P=model.param.P;
Kp=model.param.Kp;

pix=model.grid.pixel;
pho=model.grid.photo;
[defBetaX defBetaY]=deformationBeta(local,model);

defPixX=pix.X-defBetaX';
defPixY=pix.Y-defBetaY';

defPixX=diag(defPixX)*ones(P*P,Kp);
defPixY=diag(defPixY)*ones(P*P,Kp);

rpX=(diag(pho.X)*ones(Kp,P*P))';
rpY=(diag(pho.Y)*ones(Kp,P*P))';

PHI_DEF=exp(-(1/(2*SIG_PHO^2))*((defPixX-rpX).^2+(defPixY-rpY).^2));


