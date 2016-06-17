input_image = double(imread('ski_image.jpg'));
sz = size(input_image);
N = sz(1)*sz(2); 
input_image  =  reshape(input_image, N, sz(3))'/255;

%intial covariance
cov_1=eye(3);
cov_2=eye(3);
cov_3=eye(3);

%intial means 
mean_1 = [ 120; 120; 120]/255;
mean_2 = [ 12; 12; 12]/255;
mean_3 = [ 180; 180; 180]/255;

%intial mixing coefficients
Pi_1 = 1/3;
Pi_2 = 1/3;
Pi_3 = 1/3;

%reponsibilities
responsibility_1 = zeros(N,1);
responsibility_2 = zeros(N,1);
responsibility_3 = zeros(N,1);

%log likelihood
Likelihood_Matrix=[];

for j = 1:100

    %E-Step
    likelihood = 0;
    for i = 1:N
        res1 = (Pi_1*exp(-0.5*(input_image(:,i)-mean_1)'*(inv(cov_1))*(input_image(:,i)-mean_1)))/((2*pi)^1.5*det(cov_1)^.5);
        res2 = (Pi_2*exp(-0.5*(input_image(:,i)-mean_2)'*(inv(cov_2))*(input_image(:,i)-mean_2)))/((2*pi)^1.5*det(cov_2)^.5);
        res3 = (Pi_3*exp(-0.5*(input_image(:,i)-mean_3)'*(inv(cov_3))*(input_image(:,i)-mean_3)))/((2*pi)^1.5*det(cov_3)^.5);
        likelihood = likelihood+log(res1 + res2 + res3);
        responsibility_1(i) = res1/(res1 + res2 + res3);
        responsibility_2(i) = res2/(res1 + res2 + res3);
        responsibility_3(i) = res3/(res1 + res2 + res3);
    end
    Likelihood_Matrix = [Likelihood_Matrix likelihood];

    % M-step
    Nk_1 = sum(responsibility_1);
    Pi_1 = Nk_1/N;
    
    Nk_2 = sum(responsibility_2);
    Pi_2 = Nk_2/N;

    Nk_3 = sum(responsibility_3);
    Pi_3 = Nk_3/N;

    m_1 = zeros(3,1);
    m_2 = zeros(3,1);
    m_3 = zeros(3,1); 
    for k=1:N
        m_1 = m_1 + responsibility_1(k)*input_image(:,k);
        m_2 = m_2 + responsibility_2(k)*input_image(:,k);
        m_3 = m_3 + responsibility_3(k)*input_image(:,k);
    end
    mean_1 = m_1/Nk_1;
    mean_2 = m_2/Nk_2;
    mean_3 = m_3/Nk_3;


    c_1 = zeros(3,3);
    c_2 = zeros(3,3);
    c_3 = zeros(3,3);
    for i = 1:N
        c_1 = c_1+responsibility_1(i)*(input_image(:,i)-mean_1)*(input_image(:,i)-mean_1)';
        c_2 = c_2+responsibility_2(i)*(input_image(:,i)-mean_2)*(input_image(:,i)-mean_2)';
        c_3 = c_3+responsibility_3(i)*(input_image(:,i)-mean_3)*(input_image(:,i)-mean_3)';
    end
    cov_1 = c_1/Nk_1;
    cov_2 = c_2/Nk_2;
    cov_3 = c_3/Nk_3;

    if j > 1
        test = abs(Likelihood_Matrix(j) - Likelihood_Matrix(j - 1));
        if test <= (10^-5)*(Likelihood_Matrix(j - 1))
            break
        end
    end
end

[values, max_index] = max([responsibility_1  responsibility_2  responsibility_3 ], [], 2);
output_image = ones(3,N)*100;
for i=1:N
    output_image(max_index(i),i) = 0;
end
output_image = output_image';
output_image = reshape(output_image, 321, 481, 3);
figure;
imshow(output_image);

plot(1:length(Likelihood_Matrix), Likelihood_Matrix); 
grid on;
xlabel('Iterations'); 
ylabel('Log Likelihood');