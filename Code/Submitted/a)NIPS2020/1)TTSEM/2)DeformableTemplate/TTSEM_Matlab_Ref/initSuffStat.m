function suffStat=initSuffStat(param)
C=param.C;
Kp=param.Kp;
Kg=param.Kg;

suffStat.s0=zeros(C,1);
suffStat.s1=zeros(Kp,C);
suffStat.s2=zeros(Kp,Kp,C);
suffStat.s3=zeros(2*Kg,2*Kg,C);
suffStat.s4=zeros(C,1);

