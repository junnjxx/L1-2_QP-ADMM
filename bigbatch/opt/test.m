%%
rng(13)

bs=250; 
%% generate and save HOG features
% txtFilePath = '/home/ubuntu/xlj/opt/bigbatch/Datasets/MNIST_jpg/train.txt';
txtFilePath = '/home/ubuntu/xlj/opt/bigbatch/cifar10_datasets/train_gray.txt';
datasize = 50000;
filePaths = readlines(txtFilePath);
filePaths_datasize = cellstr(filePaths);
% 保留路径部分，假设路径和标签用 ',' 分隔
for i = 1:length(filePaths_datasize)
    lineData = strsplit(filePaths_datasize{i}, ','); % 按 ',' 分割
    filePaths_datasize{i} = strtrim(lineData{1});   % 只保留路径并去掉多余空格
end
filePaths_datasize = filePaths_datasize(1:datasize);
imds = imageDatastore(filePaths_datasize, 'LabelSource', 'none');
% 查看 datastore 中的文件路径
disp(imds.Files);
disp(imds);
images = readall(imds);
featureMatrix = zeros(datasize,324);
for i = 1 : datasize
    features = extractHOGFeatures(images{i});
    featureMatrix(i,:) =  features; % 添加到矩阵中
end
writematrix(featureMatrix,'cifar10_HOG_feature.csv');
disp('Image paths saved successfully');
%% write for bigbatch mnist/cifar10 need convert to gray image, datasize, targetsize
% profile on
% 定义路径文件名
% txtFilePath = '/home/ubuntu/xlj/opt/bigbatch/cifar10_datasets/train_gray_wo_label.txt';
% txtFilePath = '/home/ubuntu/xlj/opt/bigbatch/Datasets/MNIST_jpg/train.txt';
% datasize =60;
% filePaths = readlines(txtFilePath);
% filePaths_datasize = cellstr(filePaths);
% 
% % 保留路径部分，假设路径和标签用 ',' 分隔
% for i = 1:length(filePaths_datasize)
%     lineData = strsplit(filePaths_datasize{i}, ','); % 按 ',' 分割
%     filePaths_datasize{i} = strtrim(lineData{1});   % 只保留路径并去掉多余空格
% end
% 
% filePaths_datasize = filePaths_datasize(1:datasize);
% 
% imds = imageDatastore(filePaths_datasize, 'LabelSource', 'none');
% % 查看 datastore 中的文件路径
% disp(imds.Files);
% disp(imds);
% targetSize = [32 32]; % 设置目标图像大小
% images = readall(imds);
% k = 1;
% allImages = cat(4, images{(k-1)*datasize +1 : k*datasize}); % 将 cell 数组转为 4D 矩阵 (宽 x 高 x 通道 x 图片数)
% resizedImages = imresize(allImages, targetSize); % 批量调整大小
% flatImages = reshape(resizedImages, prod(targetSize), datasize)'; % 每行是一个展平的图像
% imageMatrix = flatImages;
% imagePaths = imds.Files((k-1)*datasize +1:k*datasize); % 批量提取文件路径
% % 将图像路径保存为 TXT 文件
% fileID = fopen('imagePaths_60000_60000_250_matrix_mnist.txt', 'w');
% fprintf(fileID,'%s\n', imagePaths{:});
% fclose(fileID);
% disp('Image paths saved successfully');
% profile off
% profile viewer
% profsave(profile('info'), 'profile_results_long');


% %% write for bigbatch cifar10%%%%%%%%%% 
% % profile on
% dataFolderPath = '/home/ubuntu/xlj/opt/bigbatch/datasets/train_cifar10/';
% imds = imageDatastore(dataFolderPath);
% 
% % 显示图像数据存储信息
% disp(imds);
% datasize = 10000;
% % 初始化一个空的矩阵来存储抽样图像和路径列表
% imageMatrix = zeros(datasize,1024);
% imagePaths = cell(1,datasize); % 存储选定图像的路径
% targetSize = [32 32]; % 设置目标图像大小
% 
% imds = shuffle(imds);
% images = readall(imds);
% k = 1;
% % 将图像转换为目标大小的向量并添加到矩阵中
% for i = 1 : datasize
%     j = (k-1)* datasize + i;
%     grayImage = rgb2gray(images{j});
%     resizedImage = imresize(grayImage, targetSize); % 调整图像大小
%     imageVector = reshape(resizedImage, 1, []); % 将图像展平为向量
%     imageMatrix(i,:) =  imageVector; % 添加到矩阵中
%     % 记录图像路径
%     imgfile = imds.Files{j};
%     imagePaths{i} = imgfile; % 将路径添加到列表中
% end
% 
% % 将图像路径保存为 TXT 文件
% fileID = fopen('CIFAR10_imagePaths_60000_10000_1.txt', 'w');
% % for i = 1:numel(imagePaths)
% %     fprintf(fileID, '%s\n', imagePaths{i});
% % end
% fprintf(fileID,'%s\n', imagePaths{:});
% fclose(fileID);
% disp('Image paths saved successfully');
% % profile off
% % profile viewer
% % profsave(profile('info'), 'profile_results_long');


%%
% pairwiseNorms = pdist(imageMatrix, 'euclidean');
% distanceMatrix = squareform(pairwiseNorms);
% bandwidth = sum(sum(distanceMatrix))./(datasize^2 - datasize);
% phi0 = exp(-distanceMatrix./bandwidth);
% % %%
% % 
% %% 
% vector_I = vector_method(bs,phi0);
% writematrix(vector_I,'cifar10_vector_50000_50000_250.csv');

%% 先跑 matrix的结果 
% lp_I = lp_vector(bs,data_vector);
imageMatrix = readmatrix('cifar10_HOG_feature.csv');
matrix_I = matrix_method(bs,double(imageMatrix));
writematrix(matrix_I,'cifar10_matrix_50000_250_HOG.csv');









