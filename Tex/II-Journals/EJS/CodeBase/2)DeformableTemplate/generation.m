%onlineref = load("models/refs/online_ref.mat");
%dispTemplate(onlineref.model,500,1)

%onlinescalaire= load("models/refs/modelOnlineScalaireRigide.mat"); 
%dispTemplate(onlinescalaire.modelOnlineScalaireRigide.model,500,1)

online = load("models/localruns/tmp_online.mat");
full = load("models/localruns/tmp_batch.mat");
incremental = load("models/localruns/tmp_incremental.mat");
vr = load("models/localruns/tmp_vr.mat");
fi = load("models/localruns/tmp_fi.mat");


it = 30;
%iter = [3,10,20,25,30,40];
%iter = [3,10,20,50, 80, 90];
%iter = [1,2,3,4,5,6,10,20,40,60,80,99];
iter = [1,10,20,30,40];

%dispTempIter(fi.model,iter,1)


%display All Algos inference
%models = [full,online,incremental,vr,fi];
models = [full,online,incremental,vr,fi];
dispTempAll(models,iter,1)

%Various parameters convergence plots
%disp = fi.model;
%convergence_weight(disp,it)
%convergence_sigma(disp,it,1)
%convergence_gamma(disp,it,1)

