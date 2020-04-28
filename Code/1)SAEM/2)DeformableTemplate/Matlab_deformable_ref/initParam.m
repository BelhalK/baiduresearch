function param=initParam()

%%paramètres statiques
param.C=1;
param.P=16;
param.Np=15;
param.Ng=6;
param.Kp=(param.Np)^2;
param.Kg=(param.Ng)^2;

param.SIG_PHO=0.08;%0.09%0.2;%12;%%%%%%%%%%%0.1
param.SCALE_PHO=1;
param.SIG_GEO=0.16;%0.12;%3;
param.SCALE_GEO=1;

param.sig0=0.1;

param.nbItEM=30;

param.nbData=100;

param.STOCHASTIC_STEP_PARAM=0.6;
param.burnSuffStat=0;
param.PROBA_REFRESH_LOCAL=1;
param.SIG_RANDOM_WALK_LOCAL=0.025;
param.PROBA_REFRESH_RIGID=1;
param.SIG_RANDOM_WALK_RIGID=0.01;
param.PROBA_REFRESH_HOMOT=1;
param.SIG_RANDOM_WALK_HOMOT=0.01;
param.PROBA_REFRESH_TRANS=1;
param.SIG_RANDOM_WALK_TRANS=0.01;
param.var_rigid=0.5;
param.var_trans=0.5;
param.updateSuffStat=cumsum([param.C 50 ones(1,param.nbItEM)]);
param.updateTheta=cumsum([param.C 150 ones(1,param.nbItEM)]);
param.nbItMCMC=[6*ones(1,10000)];
param.burnMCMC=[2*ones(1,10000)];
%param.nbItMCMC=[400*ones(1,10000)];
%param.burnMCMC=[300*ones(1,10000)];

%%paramètres dynamiques
param.nbItEM_completed=0;
param.acceptationMH_local=0;
param.totalMH_local=0;
param.acceptationMH_rigid=0;
param.totalMH_rigid=0;
param.acceptationMH_trans=0;
param.totalMH_trans=0;
param.acceptationMH_homot=0;
param.totalMH_homot=0;