init = 5;
iter = 10
0;
eta  = 0.1;

x_gd = zeros(iter,1); 
x_gd(1) = init;
for t=1:iter-1
	x_gd(t+1) = x_gd(t)  - eta *x_gd(t);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u    = zeros(iter,1); 
u(1) = init;
v    = zeros(iter,1); 
v(1) = init;
for t=1:iter-1
	u(t+1) = u(t)  - eta *v(t);
	v(t+1) = u(t+1)  - eta *v(t);
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y    = zeros(iter,1); 
y(1) = init;
z    = zeros(iter,1); 
z(1) = init;
GradMa = zeros(iter,1);
GradMa(1) = init;
U = zeros(iter-1,1);
m    = zeros(iter,1);
for t=1:iter-1
	y(t+1) = y(t)  - eta *z(t);
	%optm = ex(GradMa);
    if( t > 1)
        U(t-1) = GradMa(t) - GradMa(t-1);
        UU = U(1:t-1);
        %size(UU)
        ddd = UU*UU'+ 0.001*eye(size(UU*UU',1));
        cc = ones(t-1,1) \ ddd;
        cc = cc ./ sum(cc);
        m(t) = dot( cc' , GradMa(1:t-1) );
    	z(t+1) = y(t+1)  - eta *m(t);
        GradMa(t+1) = z(t+1);
    else
        z(t+1) = y(t+1);
        GradMa(t+1) = z(t+1);
    end
end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dx = abs( x_gd );
dv = abs( v );
dz = abs( z );

xData = 1:iter;
yData{1} = dx;
yData{2} = dv;
yData{3} = dz;

legendStr = {'GD','Opt-1','Opt-extra'};

%% Pretty Plot

figure;
options.logScale = 2;
options.colors = [1 0 0
    0 1 0
    0 0 1];
options.lineStyles = {':','--','-'};
options.markers = {'o','s','x'};

options.markerSpacing = [1 1
    1 1
    1 1];
options.xlabel = 'Iteration Number';
options.ylabel = 'w_t';
options.legend = legendStr;
options.legendLoc = 'NorthEast';
prettyPlot(xData,yData,options);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


m1 = v(1:end) ;
m2 = m(1:end) ;

xxData = 1:iter;
yyData{1} = v - u;
yyData{2} = z - y;

legendStr = {'Opt-1','Opt-extra'};

%% Pretty Plot

figure;
options.logScale = 0;
options.colors = [ 0 1 0
    0 0 1];
options.lineStyles = {'--','-'};
options.markers = {'s','x'};

options.markerSpacing = [1 1
    1 1
    1 1];
options.xlabel = 'Iteration Number';
options.ylabel = 'Scaled m_t';
options.legend = legendStr;
options.legendLoc = 'NorthEast';
prettyPlot(xxData,yyData,options);

