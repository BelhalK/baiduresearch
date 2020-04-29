%onlineref = load("tmp_new/online_ref.mat");
%dispTemplate(onlineref.model,500,1)

%onlinescalaire= load("tmp_new/modelOnlineScalaireRigide.mat"); 
%dispTemplate(onlinescalaire.modelOnlineScalaireRigide.model,500,1)

online = load("tmp_new/tmp_online.mat");
full = load("tmp_new/tmp_batch.mat");
incremental = load("tmp_new/tmp_incremental.mat");
vr = load("tmp_new/tmp_vr.mat");
fi = load("tmp_new/tmp_fi.mat");

it = 30;
%iter = [3,10,20,25,30,40];
%iter = [3,10,20,50, 80, 90];
%iter = [1,2,3,4,5,6,10,20,40,60,80,99];
iter = [1,2,3,4,5,6,10,20];

dispTempIter(fi.model,iter,1)


%display All Algos inference
%models = [full,online,incremental,vr,fi];
%dispTempAll(models,iter,1)

%Various parameters convergence plots
%disp = fi.model;
%convergence_weight(disp,it)
%convergence_sigma(disp,it,1)
%convergence_gamma(disp,it,1)

