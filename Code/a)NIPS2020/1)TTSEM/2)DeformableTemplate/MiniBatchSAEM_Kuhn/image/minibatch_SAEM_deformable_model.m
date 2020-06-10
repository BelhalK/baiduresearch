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


%% deformable model parameter estimation using minibatch SAEM
%% database USPOSTAL
clear
%
% data loading from USPS database digit 5
load datadig5
%
% EM  parameters
Em.nbiter=30;	%   maximal nb of EM iteration 
Em.gam=0.0001;	%   gradient coefficient
Em.precgrad=1e-04;
Em.nbgrad=15;	%  maximal nb of gradient  iteration 
Em.nbalsig=2;	% maximal nb of iteration for  alpha/sigma
Em.nbobs=20; %500 % nb of observations y_i
Em.deltaLan= 1e-03; 
Em.bLan=1000;
Em.precError = 0.0001;%EM precision
Em.minibatchsize=0.1;
%
est.Em=Em;    
%     
% Hyperparameters for learning
Hyp.imsize=data.imsize;
% Hyperparameters for photometry
% control points for Vp on a grid of size nxn
Hyp.photo.n=15;
Hyp.photo.a=200;   	%[200]	% prior weight for sigma0;
Hyp.photo.dl=1.5;		%  prototype size
Hyp.photo.I0size=40;		% corresponding image size in pixel I0sizexI0size
% Possibly larger than geo.dl for large deformations
Hyp.photo.sigma0=0.3; %[0.3]	
Hyp.photo.sigV=0.2;	% standard deviation of gaussian RKHS kernel Vp
Hyp.photo.sigalpha=1.;		% covariance Gamma^0_p: Gamma^0_p(k,k)=sigalpha^2;
Hyp.photo.k=Hyp.photo.n^2;
Hyp.photo.mu=zeros(Hyp.photo.k,1);
% Hyperparameters for geometry
% control points for Vp on a grid of size ngxng
Hyp.geo.n=6;
Hyp.geo.a=0.5;			% prior weight
Hyp.geo.dl=1;			% observation size
Hyp.geo.sigV=0.3;		% standard deviation of gaussian RKHS kernel Vg for deformations;
Hyp.geo.sigsupp=2;	% 
Hyp.geo.sigbeta=0.3;	% covariance Gamma^0_g: Gamma^0_g(k,k)=sigbeta^2;
Hyp.geo.k=Hyp.geo.n^2;
Hyp.geo.sigVreg=0.1;		% standard deviation of gaussian RKHS kernel for gradient deformation regularisation 

% 
%% geometry
dl=Hyp.geo.dl;
u=linspace(-dl,dl,Hyp.geo.n);
k=Hyp.geo.k;
[X Y]=meshgrid(u,u);
Hyp.geo.nodeX=X; Hyp.geo.nodeY=Y; 
K=exp(-((X(:)*ones(1,k)-ones(k,1)*X(:)').^2+...
   (Y(:)*ones(1,k)-ones(k,1)*Y(:)').^2)...
   /(2*Hyp.geo.sigV^2));
K=K.*exp(-((X(:)*ones(1,k)).^2+(Y(:)*ones(1,k)).^2+...
   (ones(k,1)*X(:)').^2+(ones(k,1)*Y(:)').^2)...
   /(2*Hyp.geo.sigsupp^2));
K=blkdiag(K,K);
Hyp.geo.R0=Hyp.geo.sigbeta^(-2)*K;
Hyp.geo.Gam0=inv(Hyp.geo.R0)/Hyp.geo.a;
%
%% gradient regularisation 
K=exp(-((X(:)*ones(1,k)-ones(k,1)*X(:)').^2+...
   (Y(:)*ones(1,k)-ones(k,1)*Y(:)').^2)...
   /(2*Hyp.geo.sigVreg^2));
K=K.*exp(-((X(:)*ones(1,k)).^2+(Y(:)*ones(1,k)).^2+...
   (ones(k,1)*X(:)').^2+(ones(k,1)*Y(:)').^2)...
   /(2*Hyp.geo.sigsupp^2));
K=blkdiag(K,K);
Hyp.geo.R0reg=Hyp.geo.sigbeta^(-2)*K;
%
%% photometry
%
% grid construction for geometry 
%
u=linspace(-Hyp.geo.dl,Hyp.geo.dl,Hyp.imsize);
[X,Y]=meshgrid(u,u);
Hyp.geo.refX=X; Hyp.geo.refY=Y; 
%
dl=Hyp.photo.dl; n=Hyp.photo.n;
u=linspace(-dl*(1-0.5/n),dl*(1-0.5/n),n);
[X,Y]=meshgrid(u,u);
Hyp.photo.nodeX=X; Hyp.photo.nodeY=Y;
k=Hyp.photo.k;
tmp=(X(:)*ones(1,k)-ones(k,1)*X(:)').^2;
tmp=tmp+(Y(:)*ones(1,k)-ones(k,1)*Y(:)').^2;
tmp=tmp/(2*Hyp.photo.sigV^2); 
K=exp(-tmp);
Hyp.photo.R0=Hyp.photo.sigalpha^(-2)*K;
Hyp.photo.Gam0=inv(Hyp.photo.R0);
% grid construction for photometry
%
dl=Hyp.photo.dl;
u=linspace(-dl,dl,Hyp.photo.I0size);
[X,Y]=meshgrid(u,u);
Hyp.photo.refX=X; Hyp.photo.refY=Y; 
% 
est.Hyp=Hyp;
%	  
%random seed
s = randn('state');
randn('state',s);
%       
% stochastic approximation step size
chauffe=1;
pas=[ones(1,chauffe) [1:est.Em.nbiter].^(-0.6)];
%	
%% model initialisation (parameters, interpolation kernel)
%
%  model initialisation  m_g and m_p
%  
Hyp=est.Hyp;
imsize=est.Hyp.imsize;
%
Model.photo.alpha=Hyp.photo.mu;
Model.photo.sigma{1}=Hyp.photo.sigma0;
Model.geo.Gam=Hyp.geo.Gam0;
Model.geo.R=Hyp.geo.R0;
% interpolation kernel for geometry 
%
nodeX=Hyp.geo.nodeX; nodeY=Hyp.geo.nodeY; 
X=Hyp.geo.refX; Y=Hyp.geo.refY;
k=Hyp.geo.k; 
N=imsize^2; 
K=zeros(k,N);
for i=1:k
   K(i,:)=exp(-((X(:)'-nodeX(i)).^2+(Y(:)'-nodeY(i)).^2)...
      /(2*Hyp.geo.sigV^2));
   K(i,:)=K(i,:).*exp(-((X(:)').^2+(Y(:)').^2 +nodeX(i)^2+nodeY(i)^2)...
      /(2*Hyp.geo.sigsupp^2));
end
Model.geo.K=K';
% interpolation kernel for photometry 
%
%matrix Hyp.photo.k x Hyp.photo.I0size^2
%
X=Hyp.photo.refX; Y=Hyp.photo.refY;
nodeX=Hyp.photo.nodeX; nodeY=Hyp.photo.nodeY; 
k=Hyp.photo.k; 
N=Hyp.photo.I0size^2; 
K=zeros(k,N);
for i=1:k
   K(i,:)=exp(-((X(:)'-nodeX(i)).^2+(Y(:)'-nodeY(i)).^2)...
      /(2*Hyp.photo.sigV^2));
end
Model.photo.K=K';
%
K1=zeros(k,N);
K2=K1;
for i=1:k
   K1(i,:)= 1./(2*Hyp.photo.sigV^2) * exp(-((X(:)'-nodeX(i)).^2+(Y(:)'-nodeY(i)).^2)...
      /(2*Hyp.photo.sigV^2)) .* (nodeX(i)-X(:)');
   K2(i,:)= 1./(2*Hyp.photo.sigV^2) * exp(-((X(:)'-nodeX(i)).^2+(Y(:)'-nodeY(i)).^2)...
      /(2*Hyp.photo.sigV^2)) .* (nodeY(i)-Y(:)');
end
Model.photo.K1=K1';
Model.photo.K2=K2';
%
% a priori prototype 
%
I0=Model.photo.K*Model.photo.alpha;
I0=reshape(I0,Hyp.photo.I0size,Hyp.photo.I0size);
Model.photo.I0=I0;
% Calcul of YY
%
Model.YY=0;
for i=1:est.Em.nbobs
    yi=data.Im{i};
    Model.YY=Model.YY+sum(yi(:).^2);
end
Model.YY=Model.YY/est.Em.nbobs;
%
est.Model=Model;
%
grefX=est.Hyp.geo.refX;
grefY=est.Hyp.geo.refY;
pnodeX=est.Hyp.photo.nodeX;
pnodeY=est.Hyp.photo.nodeY;
%
est.Model.sBeta=zeros(2*(est.Hyp.geo.n)^2);
est.Model.sKY = zeros((est.Hyp.photo.n)^2,1);
est.Model.sKK = zeros((est.Hyp.photo.n)^2);
%
ag=est.Hyp.geo.a;
ap=est.Hyp.photo.a;
kp=est.Hyp.photo.k;
N=est.Hyp.imsize^2;
I0size=est.Hyp.photo.I0size;
%save parameter estimation
sauvsigma=zeros(est.Em.nbiter,1);
sauvalpha=zeros(est.Em.nbiter,kp);
malpha=zeros(kp,1);
msigma=0;
%
est.betaSim= zeros(2*(est.Hyp.geo.n)^2,est.Em.nbobs);
dessin=[];
est.accept=0;
%
l=0;
%
error=est.Em.precError +1;
while (max(error) > est.Em.precError) & (l<est.Em.nbiter +1)
   l=l+1; 
   est.l=l;
   est.I0=est.Model.photo.I0;
   est.geo.Gam=est.Model.geo.Gam;
   est.geo.R=inv(est.geo.Gam);
   est.Model.sigma=est.Model.photo.sigma{l};
   %
   est.Model.KY=zeros(kp,1);
   est.Model.KK=zeros(kp);
   est.Model.BB = zeros(2*est.Hyp.geo.k);
   %
   est.Model.CholGam=chol(est.Model.geo.Gam);
   n=est.Em.nbobs;
   indice=(rand(n,1)<est.Em.minibatchsize);
   % Iteration on observations
   for i=1:est.Em.nbobs
       est.i=i;
       Im=data.Im{i};
       Im=Im(:); 
       est.Im=Im;
       if indice(i)==0 
           B=est.betaSim(:,est.i);
           else
%          % 
           est=SimulationviaGibbs(est);
           %   
           B=est.betaSim(:,est.i);
       end
       est.Model.BB=est.Model.BB+B*B'; 
       %
       est.zSimX=reshape(est.Model.geo.K*est.betaSim(1:(est.Hyp.geo.n)^2,i),est.Hyp.imsize,est.Hyp.imsize);
       est.zSimY=reshape(est.Model.geo.K*est.betaSim((est.Hyp.geo.n)^2+1:2*(est.Hyp.geo.n)^2,i),est.Hyp.imsize,est.Hyp.imsize);
       %
       est.zmX{i}=est.zSimX;
       est.zmY{i}=est.zSimY;
       %
       gdefX=grefX-est.zmX{i};
       gdefY=grefY-est.zmY{i};
       tmp=(gdefX(:)*ones(1,kp)-ones(N,1)*pnodeX(:)').^2+...
            (gdefY(:)*ones(1,kp)-ones(N,1)*pnodeY(:)').^2;
       K=exp(-tmp/(2*est.Hyp.photo.sigV^2)); 
       %
       est.Model.KK=est.Model.KK+K'*K; 
       est.Model.KY=est.Model.KY+K'*est.Im;        
   end   
   %
   n=est.Em.nbobs;
   %  sufficient statistic update
   est.Model.sBeta= est.Model.sBeta* (1- pas(l)) + pas(l)* est.Model.BB/n ;
   est.Model.sKY = est.Model.sKY* (1- pas(l)) + pas(l)* est.Model.KY/n;
   est.Model.sKK = est.Model.sKK* (1- pas(l)) + pas(l)* est.Model.KK/n ;
   %
   %  parameters update
   %
   % geometry
   est.Model.geo.Gam=(n*est.Model.sBeta+ag*est.Hyp.geo.Gam0)/(n+ag);
   est.Model.geo.R=inv(est.Model.geo.Gam);
   %
   % photometry
   %% Estimation iterative de alpha et sigma
   % Initialisation des valeurs  
   alpha=zeros(kp,1);
   sigma=1;
   % Pr�-Calcul de R0mu
   R0mu=est.Hyp.photo.R0*est.Hyp.photo.mu;    
   % iteration sur alpha et sigma
   for ll=1:1%est.Em.nbalsig
       alpha=inv(est.Model.sKK+(sigma^2/n)*est.Hyp.photo.R0)...
          *(est.Model.sKY+R0mu/n);
       sigma=sqrt((n*(est.Model.YY + alpha'*est.Model.sKK*alpha...
          -2*alpha'*est.Model.sKY)+ap*est.Hyp.photo.sigma0^2)/(n*N+ap));
   end
   sauvsigma(l)=sigma;
   msigma=(l*msigma+sigma)/(l+1);
   sauvalpha(l,:)=alpha;
   malpha=(l*malpha+alpha)/(l+1);
   est.Model.photo.alpha=alpha;
   est.Model.photo.sigma{l+1}=sigma;
   est.Model.photo.I0=reshape(est.Model.photo.K*alpha,I0size,I0size);
   %figure 1 
   %first iterations only / small nbiterem
    figure(1)
    if est.l==10
         subplot(1,3,1);
         image(90*est.Model.photo.I0'+1); colormap(gray(200)) 
    end
    if est.l==20
         subplot(1,3,2);
         image(90*est.Model.photo.I0'+1); colormap(gray(200)) 
    end
    if est.l==30
         subplot(1,3,3);
         image(90*est.Model.photo.I0'+1); colormap(gray(200)) 
    end
end %end while
%
% figure 2
%nbiterem large
% sample  N  synthetic images to evaluate  learning of geometry
%
N=20;
%
Y=zeros(2*16,16*N);
%	
sigma=est.Model.photo.sigma;
%	
I0size=est.Hyp.photo.I0size;   
I0=reshape(est.Model.photo.K*alpha,I0size,I0size);
% sample beta_gamma 
Chol1=chol(est.Model.geo.Gam);
%
for n=1:N
    %
	beta1= Chol1 * randn(size(Chol1,1),1);
	%
	%  z = Kg beta
	zX1=reshape(est.Model.geo.K*beta1(1:est.Hyp.geo.k),est.Hyp.imsize,est.Hyp.imsize);
	zY1=reshape(est.Model.geo.K*beta1(est.Hyp.geo.k+1:2*est.Hyp.geo.k),est.Hyp.imsize,est.Hyp.imsize);
	%
	%  zI0 with interpolation
	Imdef1=interp2(est.Hyp.photo.refX,est.Hyp.photo.refY,I0,...
            est.Hyp.geo.refX-zX1,est.Hyp.geo.refY-zY1,'*linear');
	Imdef2=interp2(est.Hyp.photo.refX,est.Hyp.photo.refY,I0,...
            est.Hyp.geo.refX+zX1,est.Hyp.geo.refY+zY1,'*linear');
	%
	% sample y from N(zI0,sigma^2 Id)
	y1(1:16,(n-1)*16+1:n*16)=transpose(Imdef1);
    y2(1:16,(n-1)*16+1:n*16)=transpose(Imdef2);
end
Y=[y1;y2];
%
figure(2); 
image(90*Y+1); 
colormap(gray(200)); 
axis off



         
  