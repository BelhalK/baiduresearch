function im(image,flag_figure)

dimObs=sqrt(max(size(image)));

if (flag_figure==0)
        colormap(gray),imagesc(reshape(image,dimObs,dimObs));
else
        figure,colormap(gray),imagesc(reshape(image,dimObs,dimObs));
end
 set(gca,'XTick',[],'YTick',[]);