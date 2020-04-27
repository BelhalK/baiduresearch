%modeltest = load("tmp_new/model1.mat");

%modeltest = load("tmp_new/tmp_online.mat");
%modeltest = load("tmp_new/tmp_batch.mat");
%modeltest = load("tmp_new/tmp_incremental.mat");
%modeltest = load("tmp_new/tmp_vr.mat");
modeltest = load("tmp_new/tmp_fi.mat");

modeldisp = modeltest.model;
it = 30;
flag_fig =1;
%iter = [3,10,20,25,30,40]
%iter = [3,10,20,50, 80, 100]
iter = [1,2,3,4,5,10,20,30]


%display inference
%dispTemplate(modeldisp,it,flag_fig)
dispTempIter(modeldisp,iter,flag_fig)


%Various parameters convergence plots
convergence_weight(modeldisp,it)
convergence_sigma(modeldisp,it,flag_fig)
convergence_gamma(modeldisp,it,flag_fig)