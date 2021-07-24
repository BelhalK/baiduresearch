function vr=initSuffStatVR(param)
C=param.C;
Kp=param.Kp;
Kg=param.Kg;

vr.s0=zeros(C,1);
vr.s1=zeros(Kp,C);
vr.s2=zeros(Kp,Kp,C);
vr.s3=zeros(2*Kg,2*Kg,C);
vr.s4=zeros(C,1);

