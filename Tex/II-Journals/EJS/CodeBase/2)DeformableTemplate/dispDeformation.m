function dispDeformation(beta,model,flag)

Kg=model.param.Kg;

PSI=psi(model);


defX=PSI*beta(1:Kg,:); 
defY=PSI*beta(Kg+1:end,:);

if flag
figure,quiver(model.grid.pixel.X',model.grid.pixel.Y',defX,defY);
else
    quiver(model.grid.pixel.X',model.grid.pixel.Y',defX,defY);
end