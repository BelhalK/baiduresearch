function saveModel(model)

name=model.param.name;
cd(['./tmp_new/' name]);
save('model','model');
cd('../..');