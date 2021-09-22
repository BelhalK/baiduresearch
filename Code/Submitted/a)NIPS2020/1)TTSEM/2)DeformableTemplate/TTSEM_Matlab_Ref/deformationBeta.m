function [defX defY] = deformationBeta(beta,model)


Kg=model.param.Kg;


PSI=psi(model);

    betaX=beta(1:Kg);
    betaY=beta(Kg+1:2*Kg);
    defX=PSI*betaX; 
    defY=PSI*betaY;

%figure,quiver(Rg.X',Rg.Y',betaX,betaY);
%figure,quiver(Pix.X',Pix.Y',defX,defY);




