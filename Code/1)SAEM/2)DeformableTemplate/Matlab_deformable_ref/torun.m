data = load("US_POSTAL_DIGITS_0_1.mat")
index = 5;
start = 1;
finish = 100;
data5 = data.US_POSTAL_DIGITS(start:finish,:,index);

%Online SAEM
main_function(data5)


%BATCH SAEM
start = 1;
finish = 10;
data5 = data.US_POSTAL_DIGITS(start:finish,:,index);
main_saem(data5)

%Incremental SAEM
main_isaem(data5)