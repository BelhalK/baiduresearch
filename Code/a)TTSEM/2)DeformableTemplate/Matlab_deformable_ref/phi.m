function [PHI M_p]=phi(model)

Kp=model.param.Kp;
P=model.param.P;
SIG_PHO=model.param.SIG_PHO;

Pix.X=model.grid.pixel.X;
Pix.Y=model.grid.pixel.Y;

Rp.X=model.grid.photo.X;
Rp.Y=model.grid.photo.Y;

PixX=diag(Pix.X)*ones(P*P,Kp);
PixY=diag(Pix.Y)*ones(P*P,Kp);
RpX=(diag(Rp.X)*ones(Kp,P*P))';
RpY=(diag(Rp.Y)*ones(Kp,P*P))';
dist_2=(PixX-RpX).^2+(PixY-RpY).^2;
PHI=exp(-dist_2/(2*SIG_PHO^2));

RpX=(diag(Rp.X)*ones(Kp));
RpXt=(diag(Rp.X)*ones(Kp))';
RpY=(diag(Rp.Y)*ones(Kp));
RpYt=(diag(Rp.Y)*ones(Kp))';
dist_2=(RpX-RpXt).^2+(RpY-RpYt).^2;
M_p=exp(-dist_2/(2*SIG_PHO^2));