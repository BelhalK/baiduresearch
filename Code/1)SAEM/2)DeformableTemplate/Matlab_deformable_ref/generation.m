modeltest = load("tmp_new/model1.mat");

online = load("tmp_new/tmp_online.mat");
full = load("tmp_new/tmp_batch.mat");
incremental = load("tmp_new/tmp_incremental.mat");
vr = load("tmp_new/tmp_vr.mat");
fi = load("tmp_new/tmp_fi.mat");

it = 30;
flag_fig =1;
%iter = [3,10,20,25,30,40];
%iter = [3,10,20,50, 80, 90];
iter = [1,2,3,4,5,6,10,20,40,60,80,99];

%display inference

%models = [full,online,incremental,vr,fi];
%dispTempAll(models,iter,flag_fig)


dispTemplate(modeltest.model,500,flag_fig)

%disp = fi.model;
%dispTempIter(disp,iter,flag_fig)

%Various parameters convergence plots
%convergence_weight(disp,it)
%convergence_sigma(disp,it,flag_fig)
%convergence_gamma(disp,it,flag_fig)

