online = load("models/vmruns/tmp_online.mat");
full = load("models/vmruns/tmp_batch.mat");
incremental = load("models/vmruns/tmp_incremental.mat");
vr = load("models/vmruns/tmp_vr.mat");
fi = load("models/vmruns/tmp_fi.mat");

%iter = [1,5,10,50,100,200,300,400];

models = [full,online,incremental,vr,fi];
dispTempAll(models,iter,1)


%Various parameters convergence plots
%it=100;
%disp = fi.model;
%convergence_weight(disp,it)
%convergence_sigma(disp,it,1)
%convergence_gamma(disp,it,1)

