online = load("models/vmrunsfinal/tmp_online.mat");
full = load("models/vmrunsfinal/tmp_batch.mat");
incremental = load("models/vmrunsfinal/tmp_incremental.mat");
vr = load("models/vmrunsfinal/tmp_vr.mat");
fi = load("models/vmrunsfinal/tmp_fi.mat");

vr = rmfield(vr, 'j');
fi = rmfield(fi, 'j');

%iter = [1,10,50,100,200,300,400];
%iterbatch = [1,1,1,1,2,2,3,4,5,5,5,6,7,7,8,9,9,10,10,11];
%iter = [1,10,20,30,40,50,100,150,200,210,220,230,240,250,300,350,360,370,390,400];

models = [full,online,incremental,vr,fi];


iterbatch = [1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4];
iter = [5,10,20,30,40,50,100,150,200,250,300,350,400,450,500,550,600];

dispTempAll(models,iter,iterbatch,1)

%dispOnlineRef(onlineref.model,[100,200,300,600,900,1000],1)
%dispTemplate(fi.model,800,1)

%Various parameters convergence plots
%it=100;
%disp = fi.model;
%convergence_weight(disp,it)
%convergence_sigma(disp,it,1)
%convergence_gamma(disp,it,1)

onlinescalaire= load("models/refs/modelOnlineScalaireRigide.mat"); 
%dispTemplate(onlinescalaire.modelOnlineScalaireRigide.model,500,1)