data = load("US_POSTAL_DIGITS_0_1.mat")
index = 5;
start = 1;
finish = 20;
data5 = data.US_POSTAL_DIGITS(start:finish,:,index);

%Online SAEM
main_online(data5)

%BATCH SAEM
main_saem(data5)

%Incremental SAEM
main_isaem(data5)

%VR SAEM
main_vrsaem(data5)

%FI SAEM
main_fisaem(data5)