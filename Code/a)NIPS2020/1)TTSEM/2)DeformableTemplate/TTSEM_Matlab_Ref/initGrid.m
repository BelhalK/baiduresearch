function grid=initGrid(param)

P=param.P;
Np=param.Np;
Ng=param.Ng;
Kp=param.Kp;
Kg=param.Kg;
SCALE_PHO=param.SCALE_PHO;
SCALE_GEO=param.SCALE_GEO;

pix=linspace(-1,1,P);
[pixX pixY]=meshgrid(pix,pix);
grid.pixel.X=reshape(pixX,1,P*P);
grid.pixel.Y=reshape(-pixY,1,P*P);

pho=linspace(-SCALE_PHO,SCALE_PHO,Np);
[phoX phoY]=meshgrid(pho,pho);
grid.photo.X=reshape(phoX,1,Kp);
grid.photo.Y=reshape(-phoY,1,Kp);

geo=linspace(-SCALE_GEO,SCALE_GEO,Ng);
[geoX geoY]=meshgrid(geo,geo);
grid.geo.X=reshape(geoX,1,Kg);
grid.geo.Y=reshape(-geoY,1,Kg);


