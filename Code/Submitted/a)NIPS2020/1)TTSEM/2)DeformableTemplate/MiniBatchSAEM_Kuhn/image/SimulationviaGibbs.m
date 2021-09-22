% $$$ # GPL-3.0-or-later
% $$$ # This file is part of MiniBatchSAEM
% $$$ # Copyright (C) 2020, E. Kuhn, C. Matias, T. Rebafka, S. Allassonnière, A. Trouvé
% $$$ # 
% $$$ # MiniBatchSAEM is free software: you can redistribute it and/or modify
% $$$ # it under the terms of the GNU General Public License as published by
% $$$ # the Free Software Foundation, either version 3 of the License, or any later version.
% $$$ # 
% $$$ # MiniBatchSAEM is distributed in the hope that it will be useful,
% $$$ # but WITHOUT ANY WARRANTY; without even the implied warranty of
% $$$ # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% $$$ # GNU General Public License for more details.
% $$$ # 
% $$$ # You should have received a copy of the GNU General Public License
% $$$ # along with MiniBatchSAEM.  If not, see <https://www.gnu.org/licenses/>. 

% simulation of beta 
function est=SimulationviaGibbs(est)
%
CoefLan1= reshape(est.Model.photo.K1*est.Model.photo.alpha,est.Hyp.photo.I0size,est.Hyp.photo.I0size);
CoefLan2= reshape(est.Model.photo.K2*est.Model.photo.alpha,est.Hyp.photo.I0size,est.Hyp.photo.I0size);
priorLan = - est.Model.geo.R*est.betaSim(:,est.i);
%
if est.l==1
    est.betaSim(:,est.i)=zeros(size(est.betaSim(:,est.i)));
else
%
K=est.Model.geo.K;
k=est.Hyp.geo.k;
imsize=est.Hyp.imsize;
zX=reshape(K*est.betaSim(1:k,est.i),imsize,imsize);
zY=reshape(K*est.betaSim(k+1:2*k,est.i),imsize,imsize);
dzXtmp= min(est.Hyp.geo.refX-zX, 1.5*ones(size(est.Hyp.geo.refX-zX)));
dzXtmp= max(dzXtmp, -1.5 *ones(size(est.Hyp.geo.refX-zX)));
dzYtmp= min(est.Hyp.geo.refY-zY, 1.5*ones(size(est.Hyp.geo.refY-zY)));
dzYtmp= max(dzYtmp, -1.5 *ones(size(est.Hyp.geo.refY-zY)));
%
K1def=interp2(est.Hyp.photo.refX,est.Hyp.photo.refY,CoefLan1, dzXtmp,dzYtmp,'*linear');
K2def=interp2(est.Hyp.photo.refX,est.Hyp.photo.refY,CoefLan2, dzXtmp,dzYtmp,'*linear');
%
Imdef=interp2(est.Hyp.photo.refX,est.Hyp.photo.refY,est.I0,...
    dzXtmp,dzYtmp,'*linear');
for ibeta=1:size(est.betaSim,1)/2
    mubeta(ibeta,1) = 1/2 * ( 1/est.Model.photo.sigma{est.l}^2 *...
        sum(sum( K1def .* reshape(est.Model.geo.K(:,ibeta),imsize,imsize).* reshape((Imdef(:)-est.Im),imsize,imsize)))...
        + priorLan(ibeta));
end
for ibeta=size(est.betaSim,1)/2+1:size(est.betaSim,1)
    mubeta(ibeta,1) = 1/2 * ( 1/est.Model.photo.sigma{est.l}^2 *...
        sum(sum( K2def .* reshape(est.Model.geo.K(:,ibeta-size(est.betaSim,1)/2),imsize,imsize).* ...
        reshape((Imdef(:)-est.Im),imsize,imsize))) + priorLan(ibeta));
end
%
% gradient regularisation 
%
mubetaReg(:,1) = mubeta(:,1);
%  proposal mean and variance
%
moyemp=est.betaSim(:,est.i) + min(est.Em.bLan,est.Em.deltaLan * mubetaReg);
%
varemp=(0.00001*eye(size(moyemp,1),size(moyemp,1))+...
        (mubetaReg*mubetaReg')./(max(mubetaReg.^2) ) )*est.Em.deltaLan;
%
betaTilde=moyemp+transpose(chol(varemp))*randn(size(moyemp));
%
zXTilde=reshape(K*betaTilde(1:k),imsize,imsize);
zYTilde=reshape(K*betaTilde(k+1:2*k),imsize,imsize);
dzXtmpTilde= min(est.Hyp.geo.refX-zXTilde, 1.5*ones(size(est.Hyp.geo.refX-zXTilde)));
dzXtmpTilde= max(dzXtmpTilde, -1.5 *ones(size(est.Hyp.geo.refX-zXTilde)));
dzYtmpTilde= min(est.Hyp.geo.refY-zYTilde, 1.5*ones(size(est.Hyp.geo.refY-zYTilde)));
dzYtmpTilde= max(dzYtmpTilde, -1.5 *ones(size(est.Hyp.geo.refY-zYTilde)));
%
K1defTilde=interp2(est.Hyp.photo.refX,est.Hyp.photo.refY,CoefLan1, dzXtmpTilde,dzYtmpTilde,'*linear');
K2defTilde=interp2(est.Hyp.photo.refX,est.Hyp.photo.refY,CoefLan2, dzXtmpTilde,dzYtmpTilde,'*linear');
%
ImdefTilde=interp2(est.Hyp.photo.refX,est.Hyp.photo.refY,est.I0,...
    dzXtmpTilde,dzYtmpTilde,'*linear');
%
priorLanTilde = - est.Model.geo.R*betaTilde;
%
for ibeta=1:size(est.betaSim,1)/2
    mubetaTilde(ibeta,1) = 1/2 * ( 1/est.Model.photo.sigma{est.l}^2 *...
        sum(sum( K1defTilde .* reshape(est.Model.geo.K(:,ibeta),imsize,imsize).* reshape((ImdefTilde(:)-est.Im),imsize,imsize)))...
        + priorLanTilde(ibeta));
end
for ibeta=size(est.betaSim,1)/2+1:size(est.betaSim,1)
    mubetaTilde(ibeta,1) = 1/2 * ( 1/est.Model.photo.sigma{est.l}^2 *...
        sum(sum( K2defTilde .* reshape(est.Model.geo.K(:,ibeta-size(est.betaSim,1)/2),imsize,imsize).* ...
        reshape((ImdefTilde(:)-est.Im),imsize,imsize))) + priorLanTilde(ibeta));
end
mubetaRegTilde(:,1) =mubetaTilde(:,1);
moyempTilde= betaTilde + min(est.Em.bLan, est.Em.deltaLan*mubetaRegTilde);
varempTilde=(eye(size(moyemp,1),size(moyemp,1))+...
      diag(mubetaRegTilde.^2)./(max(mubetaRegTilde.^2)-min(mubetaRegTilde.^2)) )*est.Em.deltaLan; 	
sos=gterm(est,est.betaSim(:,est.i),est.Im);
sostmp=gterm(est,betaTilde,est.Im);
%
% acceptation rule
LogSeuil =( -1/(2*(est.Model.photo.sigma{est.l})^2) * (sostmp-sos))...
    -0.5*betaTilde'*est.Model.geo.R*betaTilde+0.5*est.betaSim(:,est.i)'*est.Model.geo.R*est.betaSim(:,est.i)-...
    1/(2 ) *(  (est.betaSim(:,est.i)-moyempTilde)'* inv(varempTilde)* (est.betaSim(:,est.i)-moyempTilde) - ...
     (betaTilde-moyemp)'* inv(varemp)*(betaTilde-moyemp) ) ;
Logu=log(rand(1));
est.betaSim(:,est.i) = (LogSeuil> Logu) * betaTilde + (1- (LogSeuil> Logu)) * est.betaSim(:,est.i);
est.accept=est.accept+ (LogSeuil> Logu);
end
