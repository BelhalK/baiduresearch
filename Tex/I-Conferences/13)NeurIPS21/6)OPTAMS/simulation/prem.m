function [m] = prem(GradMa)
        %make sure GradMa >=1 2
        n = length(GradMa);
if( n >= 2)
        U = zeros( n-1, 1);
        for i=1:n-1
            U(i) = GradMa(i+1) - GradMa(i);        
        end
      
        UU = U;
        %size(UU)
        ddd = UU*UU'+ 0.001*eye(size(UU*UU',1));
        size(ddd)
        cc = ones(n-1,1) \ ddd;
        cc = cc ./ sum(cc);
        m = dot( cc' , GradMa(1:n-1) );
else
        m = 0;
end
