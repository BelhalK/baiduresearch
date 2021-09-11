function [PSI M_g]=psi(model)

P=model.param.P;
Kg=model.param.Kg;
SIG_GEO=model.param.SIG_GEO;


PixX=diag(model.grid.pixel.X)*ones(P*P,Kg);
PixY=diag(model.grid.pixel.Y)*ones(P*P,Kg);

RgX=(diag(model.grid.geo.X)*ones(Kg,P*P))';
RgY=(diag(model.grid.geo.Y)*ones(Kg,P*P))';

PSI=exp(-((PixX-RgX).^2+(PixY-RgY).^2)/(2*SIG_GEO^2));

RgX=(diag(model.grid.geo.X)*ones(Kg));
RgY=(diag(model.grid.geo.Y)*ones(Kg));
RgXt=(diag(model.grid.geo.X)*ones(Kg))';
RgYt=(diag(model.grid.geo.Y)*ones(Kg))';
dist_2=(RgX-RgXt).^2+(RgY-RgYt).^2;
M_g=exp(-dist_2/(2*SIG_GEO^2));
M_g=[M_g zeros(Kg)
zeros(Kg) M_g];