init = 1;
iter = 20

eta  = 0.4;

x_gd = zeros(iter,1); 
x_gd(1) = init;
for t=1:iter-1
    if( mod(t,3) == 1)
    	x_gd(t+1) = x_gd(t)  - eta * 3 / sqrt(t);
        x_gd(t+1) = thre(x_gd(t+1));
    else
    	x_gd(t+1) = x_gd(t)  + eta * 1 / sqrt(t);
        x_gd(t+1) = thre(x_gd(t+1));
    end    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u    = zeros(iter,1); 
u(1) = init;
v    = zeros(iter,1); 
v(1) = init;
for t=1:iter-1
    if( mod(t,3) == 1)
        u(t+1) = u(t)  - eta *3 / sqrt(t);
        u(t+1) = thre(u(t+1));
   	    v(t+1) = u(t+1)  + eta *1 / sqrt(t+1);
        v(t+1) = thre(v(t+1));
    elseif( mod(t,3) == 2)
        u(t+1) = u(t)    +  eta *1 / sqrt(t);
        u(t+1) = thre(u(t+1));
   	    v(t+1) = u(t+1)  -  eta *3 / sqrt(t+1);
        v(t+1) = thre(v(t+1));        
    else    
        u(t+1) = u(t)   + eta *1 / sqrt(t);
        u(t+1) = thre(u(t+1));
   	    v(t+1) = u(t+1) + eta *1 / sqrt(t+1);
        v(t+1) = thre(v(t+1));
    end
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y    = zeros(iter,1); 
y(1) = init;
z    = zeros(iter,1); 
z(1) = init;
GradMa = zeros(iter,1);
GradMa(1) = 3;
U = zeros(iter-1,1);
m    = zeros(iter,1);
for t=1:iter-1
    if( mod(t,3) == 1)
    	y(t+1) = y(t)  - eta *3 / sqrt(t);
        y(t+1) = thre(y(t+1));
    else
    	y(t+1) = y(t)  + eta * 1 / sqrt(t);
        y(t+1) = thre(y(t+1));
    end    
    if( t > 1) 
        cc = 3;
        m(t)   = prem(GradMa(1:t-1) );
        aaa = eta *m(t) / sqrt(t+1)
        z(t+1) = y(t+1)  - eta *m(t) / sqrt(t+1);
        z(t+1) = thre(z(t+1));
    else
        z(t+1) = y(t+1);    
    end    
    
    if( mod(t,3) == 1)
        GradMa(t+1) = 3;
    elseif ( mod(t,3) == 1)
        GradMa(t+1) = -1
    end    
        
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dx = abs( x_gd+1);
dv = abs( v+1 );
dz = abs( z+1 );

xData = 1:iter;
yData{1} = dx;
yData{2} = dv;
yData{3} = dz;

legendStr = {'GD','Opt-1','Opt-extra'};

%% Pretty Plot

figure;
options.logScale = 0;
options.colors = [1 0 0
    0 1 0
    0 0 1];
options.lineStyles = {':','--','-'};
options.markers = {'o','s','x'};

options.markerSpacing = [1 1
    1 1
    1 1];
options.xlabel = 'Iteration Number';
options.ylabel = 'Distance to the optimal point -1';
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



