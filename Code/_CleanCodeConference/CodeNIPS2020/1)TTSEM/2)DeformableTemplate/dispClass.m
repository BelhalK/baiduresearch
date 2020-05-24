function dispClass(carlinChib,it,flag_fig)
if flag_fig
    figure,hold on
end
for n=1:it
    plot(n,carlinChib(n).class,'.');
end