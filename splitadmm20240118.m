% %%
% n = 80;
% k = 10;
% A = randn(n);
% A = A*A';
% G = randn(n,k);
% % Y = splitadmm20240118(A,G);
% Y = splitadmm20240119(A,G);
%%
% % profile on
% dataFolderPath = '/home/ubuntu/xlj/optimization/Datasets/MNIST_jpg_01/';
dataFolderPath = "F:\RES\code\gray_image"
% 创建 imageDatastore，递归读取图像
imds = imageDatastore(dataFolderPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
images = readall(imds);
labels = imds.Labels;
imageMatrix = [];
% 设置目标图像大小
targetSize = [28 28];
for i = 1:numel(images)
 % for i = 1:6000
    % 将图像转换为灰度图像
%      grayImage = rgb2gray(images{i});
    grayImage = images{i};
    resizedImage = grayImage;
%     resizedImage = imresize(grayImage, targetSize);
    imageVector = reshape(resizedImage, 1, []);
    imageMatrix = [imageMatrix; imageVector];
end
% imageMatrix 包含所有图像的向量表示，每行代表一张图像
%%
n = size(imageMatrix,1);
pairwiseNorms = pdist(imageMatrix, 'euclidean');
numImages = size(imageMatrix, 1);
distanceMatrix = squareform(pairwiseNorms);
% disp(distanceMatrix);
%% 计算这个矩阵的均值/ 去除对角线 / 计算bandwidth / 计算 phi0
bandwidth = sum(sum(distanceMatrix))./(n^2 - n)
% bandwidth =1;
% 计算原始的 大phi矩阵
phi0 = exp(-distanceMatrix./bandwidth);
[n , n] = size(phi0);
bs =3;
nb = floor(n / bs)
k = n
n = nb
G = phi0 * ones(k,n);
%%   根据一定的规则增大beta和eta
A = phi0(1:(n*bs), 1:(n*bs)) ;
A = 2.*A./bs./bs;
G = G(1:(n*bs),:);
G = -4 .* G ./bs ./n;
[n_,k_] = size(G);
X = rand(n_,k_);
Y = rand(n_,k_);
sa_obj = [];
sa_error = [];
n_iter =3;
iter = 1e5;
beta = 1000;  % |X-Y|
eta = 0.1 % SUM(Y)
for i = 1:n_iter
   
    [X,Y, error_list] = splitadmm20240119(A,G,eta,beta,iter, X,Y);
    
    has_non_zero_or_one = any(Y(:) ~= 0 & Y(:) ~= 1);
    if has_non_zero_or_one == 1
        eta = eta * 10;
    end
    threshold = 0.1; % 设置阈值
    % is_converged = check_convergence(saved_obj, threshold);
    
    % if is_converged ==0
    %     beta  = beta *2;
    % end
    sa_error = [sa_error error_list] ;
    % sa_obj = [sa_obj saved_obj];
end

subplot(2, 1, 1); % 第一个子图
[u_, len] = size(sa_error);
plot_x = [1:len];
plot(plot_x, plot_x);
%semilogy(plot_x, sa_obj);

xlabel('Iterations');
ylabel('Objective Value');
title('Objective Value');


subplot(2, 1, 2); % 第二个子图

plot_x = [1: len];

%sa_error(sa_error > 20) =20;

%plot(plot_x,log(sa_error+1) );
semilogy(plot_x, sa_error);
xlabel('Iterations');

ylabel('KKT Value'); 
title('KKT Value');

%% 转化Y为batch
M = [];
for i = 1:n
    m_temp = find(any(Y(:,i), 2))';
    M = vertcat(M,m_temp );
end
M_ = M';
Sample_order = M_(:);
save('matrix_problem.mat');
csvwrite('M_mnist_01_sample_order.csv',Sample_order);
% p = profile("info");
% profsave(p,'profile_result_0407') 
function is_converged = check_convergence(sequence, threshold)
    % 计算序列的标准差
    std_dev = std(sequence(end-3:end));
    
    % 检查标准差是否小于阈值
    is_converged = std_dev < threshold;
end