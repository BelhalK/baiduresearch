function model=initModel()

model.param=initParam();
model.grid=initGrid(model.param);
model.theta=initTheta(model.param);

end