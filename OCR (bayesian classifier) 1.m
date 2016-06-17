%
% Ankit Kumar wagadre 130108026
% MANVENDRA SINGH NARWAR 130121016
%

% all have seperate covariance matrix

train_data_folder = 'C:\Users\ankit wagadre\Desktop\prml code\Assignment_list\TrainCharacters\TrainCharacters';
test_data_folder = 'C:\Users\ankit wagadre\Desktop\prml code\Assignment_list\TestCharacters\TestCharacters\TestCharacters\';

E = []; C = []; I = [];
scale_factor = 0.25; %(32/128)

for k = 1:200
    img = double(imread([train_data_folder '\1\' num2str(k) '.jpg']));
    img = imresize(img, scale_factor);
    img = reshape(img, 1024, []);
    E = [E img];
    img = double(imread([train_data_folder '\2\' num2str(k) '.jpg']));
    img = imresize(img, scale_factor);
    img = reshape(img, 1024, []);
    C = [C img];
    img = double(imread([train_data_folder '\3\' num2str(k) '.jpg']));
    img = imresize(img, scale_factor);
    img = reshape(img, 1024, []);
    I = [I img];
end

mean_E = zeros(1024,1);
mean_C = zeros(1024,1);
mean_I = zeros(1024,1);
for k=1:200
    mean_E = mean_E + E(:, k);
    mean_C = mean_C + C(:, k);
    mean_I = mean_I + I(:, k);
end
mean_E = mean_E/200;
mean_C = mean_C/200;
mean_I = mean_I/200;


covariance_E = zeros(1024,1024);
covariance_C = zeros(1024,1024);
covariance_I = zeros(1024,1024);
for m=1:200
    covariance_E = covariance_E + ((E(:, m)-mean_E)*(E(:, m)-mean_E)');  
    covariance_C = covariance_C + ((C(:, m)-mean_C)*(C(:, m)-mean_C)');  
    covariance_I = covariance_I + ((I(:, m)-mean_I)*(I(:, m)-mean_I)');
end
covariance_E = covariance_E/200;
covariance_C = covariance_C/200;
covariance_I = covariance_I/200;

regularization_const=0.25;

cov_E = covariance_E + regularization_const*eye(1024);
cov_E_inv = inv(cov_E);
cov_E_det = det(cov_E);

cov_C = covariance_C + regularization_const*eye(1024);
cov_C_inv = inv(cov_C);
cov_C_det = det(cov_C);

cov_I = covariance_I + regularization_const*eye(1024);
cov_I_inv = inv(cov_I);
cov_I_det = det(cov_I);

disc_func = zeros(3,1);
prior = 1/3; % same for all
Classification = zeros(3, 100);
for i = 1:3
    for j = 1:100
        test_img = double(imread([test_data_folder  num2str(i) '\' num2str(j + 200) '.jpg']));
        test_img = imresize(test_img,scale_factor);
        test_img = reshape(test_img,1024,1);
        disc_func(1) = -0.5*log(cov_E_det)-.5*(test_img-mean_E)'*(cov_E_inv)*(test_img-mean_E);
        disc_func(2) = -0.5*log(cov_C_det)-.5*(test_img-mean_C)'*(cov_C_inv)*(test_img-mean_C);
        disc_func(3) = -0.5*log(cov_I_det)-.5*(test_img-mean_I)'*(cov_I_inv)*(test_img-mean_I);
        
      [~,maxindex]=max(disc_func);
      if maxindex == i
          Classification(i, j) = 1;
      else
          Classification(i, j) = 0;
      end
    end
end
