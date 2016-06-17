%
% Ankit Kumar wagadre 130108026
% MANVENDRA SINGH NARWAR 130121016
%

clear all;
clc;
noOfImages = 200;
A = zeros(200, 10304);

%file read code
for i = 1:200
    fileNo = mod(i, 5);
    if fileNo == 0
        fileNo = 5;
    end
    
    folderNo = floor((i-1)/5) + 1;
    fileAddress =  strcat('s', num2str(folderNo), '/',  num2str(fileNo), '.pgm' );
    img = double(imread(fileAddress));
    A(i, :) = reshape(img, 1, []);
end

mx = mean(A);

%covariance matrix
X_mx = zeros(200, 10304);

for i = 1:200
    X_mx(i, :) = A(i, :)-mx;
end

%cov_matrix = X_mx*X_mx';
cov_matrix = (1/200)*X_mx*X_mx';

[V,D] = eig(cov_matrix);

%part <i>
for i = 1:5
    subplot(1, 5, i);
    ev = X_mx'*V(:,i);
    ev = ev/norm(ev);
    ev_img = reshape(ev, [112 92]);
    ev_img = mat2gray(ev_img);
    imshow(ev_img)
end
V = X_mx'*V;

%part <ii> a
variace_ratio = zeros(201,1);
eigen_sum = trace(D);
for i = 2:201
    variace_ratio(i) = variace_ratio(i-1) + D(i-1,i-1);
end
figure;
cov_curve = variace_ratio*100/eigen_sum;
plot(0:200, cov_curve);

%part <ii> b
cov_curve = cov_curve > 95;
noOfDim = find(cov_curve);
display(strcat('minimum number of dimensions required for projecting the face vectors == ', num2str(noOfDim(1))));

img = double(imread('face_input_1.pgm'));
X3 = reshape(img, 1, []);
X3_mx = X3 - mx;
coef3 = X3_mx*V;
X3_out = zeros(10304, 200);
X3_out(:, 1) = coef3(1)*V(:,1) + mx';
for i = 2:200
X3_out(:, i) = X3_out(:, i-1) + coef3(i)*V(:,i);
end

error3 = zeros(1, 200);
for i = 1:200
    img_arr = mat2gray(X3_out(:, i));
    img_arr = (X3' - img_arr)/255;
    error3(1,i) = sum(img_arr.*img_arr)/10304;
end
figure;
plot(error3);

figure;
j = 1;
for i = [1 15 200]
    subplot(1, 5, j);
    img_out = reshape(X3_out(:, i), [112 92]);
    img_out = mat2gray(img_out);
    imshow(img_out);
    j = j + 1;
end


img = double(imread('face_input_2.pgm'));
X4 = reshape(img, 1, []);
X4_mx = X4 - mx;
coef4 = X4_mx*V;
X4_out = zeros(10304, 200);
X4_out(:, 1) = coef4(1)*V(:,1) + mx';
for i = 2:200
X4_out(:, i) = X4_out(:, i-1) + coef4(i)*V(:,i);
end

error4 = zeros(1, 200);
for i = 1:200
    img_arr = mat2gray(X4_out(:, i));
    img_arr = (X4' - img_arr)/255;
    error4(1,i) = (img_arr'*img_arr)/10304;
end
figure;
plot(error4);

figure;
j = 1;
for i = [1 15 100 200]
    subplot(1, 4, j);
    img_out = reshape(X4_out(:, i), [112 92]);
    img_out = mat2gray(img_out);
    imshow(img_out);
    j = j + 1;
end