%% pro7 using new algorithm
dataFolderPath ='F:\\RES\\code\\gray_image\\' ;
imds = imageDatastore(dataFolderPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
images = readall(imds);
labels = imds.Labels;
imageMatrix = [];
targetSize = [28 28];
% 将每张图像转换为向量并添加到矩阵中
for i = 1:numel(images)
% for i =1:6000
    % 将图像转换为灰度图像
    % grayImage = rgb2gray(images{i});
    grayImage = images{i};
    resizedImage = imresize(grayImage, targetSize);
    imageVector = reshape(resizedImage, 1, []);
    imageMatrix = [imageMatrix; imageVector];
end
% imageMatrix 包含所有图像的向量表示，每行代表一张图像
n = size(imageMatrix,1);
pairwiseNorms = pdist(imageMatrix, 'euclidean');
numImages = size(imageMatrix, 1);
distanceMatrix = squareform(pairwiseNorms);
%% 计算这个矩阵的均值/ 去除对角线 / 计算bandwidth / 计算 phi0
bandwidth = sum(sum(distanceMatrix))./(n^2 - n)
% 计算原始的 大phi矩阵
phi0 = exp(-distanceMatrix./bandwidth);
[n , n] = size(phi0);
bs = 3
nb = floor(n / bs )
%%
D = [1: nb*bs ];
save_B = [];
Q = D;
B = [];
P = [];
saved_Y = [];
for i = 1:(nb-1)
    P = sort(vertcat(P, Q(B)));
    Q =  sort(setdiff(Q, Q(B)));
    np = size(P,2);
    nq = size(Q,2);
    phi1 = phi0(Q,Q);
    phi2 = sum(phi0(Q,Q)).*((np+bs)/n);
    phi3 = sum(phi0(Q,P)').*((nq-bs)/n);
    Z = phi1;
    % 创建 Z
    for j = 1:nq
        if np >0
            Z(j,j) = Z(j,j) - phi2(j) + phi3(j);
        else
            Z(j,j) = Z(j,j) - phi2(j);
        end
    end
   
    n_iter = 3;
    iter = 1e5;
    X=rand(nq,1);
    Y = rand(nq,1);
    eta =1;
    beta = 1000;
    all_obj = [];
    all_error = [];
    for j = 1:n_iter
        [X,Y,sa_obj,sa_error] =splitadmm20240307(Z,nq, bs,eta,beta,iter,X,Y ); 
        has_non_zero_or_one = any(Y(:) ~= 0 & Y(:) ~= 1);
        if has_non_zero_or_one == 1
            eta = eta * 2;
        end
%         threshold = 0.1; % 设置阈值
%         is_converged = check_convergence(sa_obj, threshold);
% 
%         if is_converged ==0
%             beta  = beta *2;
%         end

        
%         all_obj = [all_obj sa_obj];
%         all_error = [all_error sa_error];
        
    end
    B = find(Y==1);
    s_B = Q(B);
    save_B = [save_B;s_B];

end
%% 最后一个batch，不需要进入优化程序
left_Q = sort(setdiff(Q, s_B));
save_B =[save_B;left_Q];
% saved_Y = saved_Y'

function is_converged = check_convergence(sequence, threshold)
    % 计算序列的标准差
    std_dev = std(sequence(end-3:end));
    
    % 检查标准差是否小于阈值
    is_converged = std_dev < threshold;
end