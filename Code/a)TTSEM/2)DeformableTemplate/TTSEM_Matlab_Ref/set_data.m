load('US_POSTAL_DIGITS_0_1.mat')
data=[US_POSTAL_DIGITS(1:20,:,1)
US_POSTAL_DIGITS(1:20,:,2)
US_POSTAL_DIGITS(1:20,:,3)
US_POSTAL_DIGITS(1:20,:,4)
US_POSTAL_DIGITS(1:20,:,5)];
data(61,:)=data(68,:);
data(63,:)=data(74,:);
for n=101:3000
di=randi(5);
nu=randi(1100);
data(n,:)=US_POSTAL_DIGITS(nu,:,di);
end