%
% Ankit Kumar wagadre 130108026
% MANVENDRA SINGH NARWAR 130121016
%

clear all;
clc;
load('Pattern1.mat')
load('Pattern2.mat')
load('Pattern3.mat')

load('Test1.mat')
load('Test2.mat')
load('Test3.mat')

train_data = zeros(600, 120);
group = zeros(600, 1);
for i = 1:200
    train_data(i, :) = train_pattern_1{1, i};
    group(i,1) = 1;
    train_data(i+200, :) = train_pattern_2{1, i};
    group(i+200,1) = 2;
    train_data(i+400, :) = train_pattern_3{1, i};
    group(i+400,1) = 3;
end

test_data = zeros(300, 120);
test_group = zeros(300, 1);
for i = 1:100
    test_data(i, :) = test_pattern_1{1, i};
    test_group(i,1) = 1;
    test_data(i+100, :) = test_pattern_2{1, i};
    test_group(i+100,1) = 2;
    test_data(i+200, :) = test_pattern_3{1, i};
    test_group(i+200,1) = 3;
end

Acc = zeros(50,1);
C = 2^(-5);
g = 2^(-15);
for i = 1:50
    options = [ '-c ', num2str(C),' -g  ', num2str(g) ]
    svmStruct = svmtrain(group, train_data, options);
    [predict_label, accuracy, prob_estimates] = svmpredict(test_group, test_data, svmStruct);
    Acc(i, 1) = accuracy(1);
    C = C*4;
    g = g*4;
end

plot(1:50, Acc);
xlabel(' Cost (2^-5, 2^-3, 2^-1 ...)  and Gamma (2^-15, 2^-13, 2^-11 ...)'); 
ylabel('Accuracy');