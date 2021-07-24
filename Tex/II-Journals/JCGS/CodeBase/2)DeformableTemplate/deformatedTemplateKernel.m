function PHI_DEF=deformatedTemplateKernel(local,rigid,trans,homot,model)

pix=model.grid.pixel;
pho=model.grid.photo;
Kg=model.param.Kg;
Kp=model.param.Kp;
SIG_PHO=model.param.SIG_PHO;
P=model.param.P;

PSI=psi(model);

defX=PSI*local(1:Kg);
defY=PSI*local(Kg+1:2*Kg);
angle=rigid(1);
center_r=rigid(2:3);
transX=trans(1);
transY=trans(2);
scale=homot(1);
center_h=homot(2:3);

tmpX=scale*(cos(angle)*(pho.X'-center_r(1)*ones(Kp,1))-sin(angle)*(pho.Y'-center_r(2)*ones(Kp,1))...
    +center_r(1)*ones(Kp,1)+transX-center_h(1))+center_h(1);
tmpY=scale*(sin(angle)*(pho.X'-center_r(1)*ones(Kp,1))+cos(angle)*(pho.Y'-center_r(2)*ones(Kp,1))...
    +center_r(2)*ones(Kp,1)+transY-center_h(2))+center_h(2);

PHI_DEF=exp(-((diag(pix.X'-defX)*ones(P*P,Kp)-(diag(tmpX)*ones(Kp,P*P))').^2+...
    (diag(pix.Y'-defY)*ones(P*P,Kp)-(diag(tmpY)*ones(Kp,P*P))').^2)/(2*SIG_PHO^2));