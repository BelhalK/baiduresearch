function dispGrid(model,arg1,type1,arg2,type2,arg3,type3)
if nargin==3
    x=['model.grid.' arg1 '.X'];
    y=['model.grid.' arg1 '.Y'];
    figure,plot(eval(x),eval(y),type1);
elseif nargin==5
    x1=['model.grid.' arg1 '.X'];
    y1=['model.grid.' arg1 '.Y'];
    x2=['model.grid.' arg2 '.X'];
    y2=['model.grid.' arg2 '.Y'];
    figure,plot(eval(x1),eval(y1),type1);
    hold on;
    plot(eval(x2),eval(y2),type2);
elseif(nargin==7)
    x1=['model.grid.' arg1 '.X'];
    y1=['model.grid.' arg1 '.Y'];
    x2=['model.grid.' arg2 '.X'];
    y2=['model.grid.' arg2 '.Y'];
    x3=['model.grid.' arg3 '.X'];
    y3=['model.grid.' arg3 '.Y'];
    figure,plot(eval(x1),eval(y1),type1);
    hold on
    plot(eval(x2),eval(y2),type2);
    plot(eval(x3),eval(y3),type3);
end