function fi=initSuffStatFI(param)
C=param.C;
Kp=param.Kp;
Kg=param.Kg;

fi.s0=zeros(C,1);
fi.s1=zeros(Kp,C);
fi.s2=zeros(Kp,Kp,C);
fi.s3=zeros(2*Kg,2*Kg,C);
fi.s4=zeros(C,1);

