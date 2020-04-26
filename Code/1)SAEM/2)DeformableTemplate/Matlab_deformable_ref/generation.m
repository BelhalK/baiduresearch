%modeltest = load("tmp_new/model1.mat");
modeltest = load("tmp_new/tmp_batch.mat");
modeldisp = modeltest.model;
it = 7;
flag_fig =1;
%display inference
dispTemplate(modeldisp,it,flag_fig)


%Various parameters convergence plots
convergence_weight(modeldisp,it)
convergence_sigma(modeldisp,it,flag_fig)
convergence_gamma(modeldisp,it,flag_fig)