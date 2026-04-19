%%
rng(13)
n= 100; % dimension % number of samples
dens=4e-3; % density of each Q 
K = 4 ; 
 K_N = [21,19,20,20]; m = sum(K_N); % K kinds , N samples in each kind
 % K_N = [94,106,100,100]; m = sum(K_N); 
% K_N = [280,320,300,300]; m = sum(K_N); 
% K_N = [4,6,5,5]; m = sum(K_N); % 测试用
% K_N = [1,0,2,1]; m = sum(K_N); % 测试用
Q = cell(1,m); G= zeros(n,m);
% data_vector = zeros(m, n*n+n);
% data_vector = zeros(m, n+n);  
data_vector = zeros(m, n);
% data_vector = zeros(m, K);
% data_vector = zeros(m, n*n);
Qa = zeros(n);
ind = 1;
% xhat = randn(n,1);
for k = 1:K
xhat = rand(n,1)  ;
% Gk = rand(n,m)-rand(1);
rc = rand(n,1) ;
% rc = rand(n,1);
Qk = sprandsym(n,dens,rc);
% [fi,fj,fv]=find(Qk);
Gk = - Qk * xhat;
% Gk = rand(n,1);
% Gk =  -(rand(n,1) * 0.02  + ((k-1)*(k+1)*0.1));
% Gk = [];
for i=1:K_N(k)
    Q{ind} = Qk;
    % Q{ind} = Qk;
    Qa = Qa + Q{ind};
    G(:,ind) = Gk .*(1 + randn(n,1)*1e-2);
    % data_vector(ind,:) = [Q{ind}(:)' G(:,ind)' ];
    % data_vector(ind,:) = [Qk(:)'];
    % data_vector(ind,:) = [rc' G(:,ind)' ];
    data_vector(ind,:) = [-Qk\G(:,ind) ];
    t_ = zeros(1,K);
    t_(k) = 1;
    % data_vector(ind,:) = t_;
    ind = ind+1;
end
end
xstar = -Qa\sum(G,2); 
fstar = xstar'*Qa*xstar/2 + sum(G,2)'*xstar;
bs= 4; % batch size 40

%% 先跑 matrix的结果 
% lp_I = lp_vector(bs,data_vector);
matrix_I = matrix_method(bs,data_vector);

vector_I = vector_method(bs,data_vector);
% matrix_I = [4 5 1 3 2; 10 6 7 8 9; 11 13 15 14 12; 16 18 19 20 17]
%%
nb=floor(m/bs); % number of batches 50
nepoch=400; % number of epoch
stepsize = 0.5/max(eig(Qa))/1;
thres = 0.2;
x = rand(n,1);

gamma = 0.0005;lamb = 0.0005;
%% 对比算法 random matrix well orth 
opt_type = 3; % SGD, WSGD, MWSGD, ADAM 
% [hist_well, well_I] = wellbatch(x,Q,G,stepsize,bs,nepoch,Qa,K,N,opt_type,gamma,lamb);
[hist_random1, hist_random1_x] = randbatch(x,Q,G,stepsize,bs,nepoch,Qa,opt_type,gamma,lamb,xstar);
[hist_random2, hist_random2_x] = randbatch(x,Q,G,stepsize,bs,nepoch,Qa,opt_type,gamma,lamb,xstar);
[hist_random3, hist_random3_x]= randbatch(x,Q,G,stepsize,bs,nepoch,Qa,opt_type,gamma,lamb,xstar);
hist_random = (hist_random1 + hist_random2 + hist_random3) /3 ;
hist_random_x = (hist_random1_x + hist_random2_x + hist_random3_x) /3 ;
[historth,Lencliq] = orthbatch(x,Q,G,stepsize,bs,nepoch,Qa,thres,opt_type,gamma,lamb);
[hist_matrix, hist_matrix_x] = generalbatch(x,Q,G,stepsize,bs,nepoch,Qa,data_vector,matrix_I,opt_type,gamma,lamb,xstar);
[hist_vector, hist_vector_x]= generalbatch(x,Q,G,stepsize,bs,nepoch,Qa,data_vector,vector_I,opt_type,gamma,lamb,xstar);
% figure; semilogy(0:nepoch, hist_random-fstar,0:nepoch, historth-fstar);legend('randbatch','orthbatch')
% figure; semilogy(0:nepoch, hist_random-fstar,0:nepoch, historth-fstar, 0:nepoch, hist_matrix-fstar);legend('randbatch','orthbatch','matrixbatch')
% figure; plot(1:nepoch,Lencliq);legend('Lencliq')
% figure; semilogy(0:nepoch, hist_random-fstar,0:nepoch, hist_matrix-fstar);legend('randbatch','matrixbatch')
% figure; plot(0:nepoch, hist_random+norm(G,'fro')^2,0:nepoch, hist_matrix+norm(G,'fro')^2);legend('randbatch','matrixbatch')
%%
% % 记录error
% % 这是random的
% error_f_random = log(hist_random-fstar);
% random_f_epoch100 = min(error_f_random(1:100));
% random_f_epoch200 = min(error_f_random(101:200));
% random_f_epoch300 = min(error_f_random(201:300));
% random_f_epoch400 = min(error_f_random(301:400));
% fprintf('random_f_epoch100:%6.2f,random_f_epoch200:%6.2f,random_f_epoch300:%6.2f,random_f_epoch400:%6.2f\n',random_f_epoch100,random_f_epoch200,random_f_epoch300,random_f_epoch400)
% error_x_random = log(hist_random_x );
% random_x_epoch100 = min(error_x_random(1:100));
% random_x_epoch200 = min(error_x_random(101:200));
% random_x_epoch300 = min(error_x_random(201:300));
% random_x_epoch400 = min(error_x_random(301:400));
% fprintf('random_x_epoch100:%6.2f,random_x_epoch200:%6.2f,random_x_epoch300:%6.2f,random_x_epoch400:%6.2f\n',random_x_epoch100,random_x_epoch200,random_x_epoch300,random_x_epoch400)
% % 这是matrix的
% error_f_matrix = log(hist_matrix-fstar);
% matrix_f_epoch100 = min(error_f_matrix(1:100));
% matrix_f_epoch200 = min(error_f_matrix(101:200));
% matrix_f_epoch300 = min(error_f_matrix(201:300));
% matrix_f_epoch400 = min(error_f_matrix(301:400));
% fprintf('matrix_f_epoch100:%6.2f,matrix_f_epoch200:%6.2f,matrix_f_epoch300:%6.2f,matrix_f_epoch400:%6.2f\n',matrix_f_epoch100,matrix_f_epoch200,matrix_f_epoch300,matrix_f_epoch400)
% error_x_matrix = log(hist_matrix_x );
% matrix_x_epoch100 = min(error_x_matrix(1:100));
% matrix_x_epoch200 = min(error_x_matrix(101:200));
% matrix_x_epoch300 = min(error_x_matrix(201:300));
% matrix_x_epoch400 = min(error_x_matrix(301:400));
% fprintf('matrix_x_epoch100:%6.2f,matrix_x_epoch200:%6.2f,matrix_x_epoch300:%6.2f,matrix_x_epoch400:%6.2f\n',matrix_x_epoch100,matrix_x_epoch200,matrix_x_epoch300,matrix_x_epoch400)
% % 这是vector的
% error_f_vector = log(hist_vector-fstar);
% vector_f_epoch100 = min(error_f_vector(1:100));
% vector_f_epoch200 = min(error_f_vector(101:200));
% vector_f_epoch300 = min(error_f_vector(201:300));
% vector_f_epoch400 = min(error_f_vector(301:400));
% fprintf('vector_f_epoch100:%6.2f,vector_f_epoch200:%6.2f,vector_f_epoch300:%6.2f,vector_f_epoch400:%6.2f\n',vector_f_epoch100,vector_f_epoch200,vector_f_epoch300,vector_f_epoch400)
% error_x_vector = log(hist_vector_x );
% vector_x_epoch100 = min(error_x_vector(1:100));
% vector_x_epoch200 = min(error_x_vector(101:200));
% vector_x_epoch300 = min(error_x_vector(201:300));
% vector_x_epoch400 = min(error_x_vector(301:400));
% fprintf('vector_x_epoch100:%6.2f,vector_x_epoch200:%6.2f,vector_x_epoch300:%6.2f,vector_x_epoch400:%6.2f\n',vector_x_epoch100,vector_x_epoch200,vector_x_epoch300,vector_x_epoch400)
%% 创建图形
figure; 
semilogy(0:nepoch, hist_random-fstar, '-^', 'LineWidth', 1, 'MarkerSize',3,'Color', [5/255,5/255,5/255]); hold on;
% semilogy(0:nepoch, historthq a-fstar, '-s', 'LineWidth', 2, 'MarkerSize', 6);
semilogy(0:nepoch, hist_matrix-fstar, '-s', 'LineWidth', 1, 'MarkerSize', 3, 'Color', [255/255,17/255,14/255]);
% semilogy(0:nepoch, hist_well-fstar, '-*', 'LineWidth', 2, 'MarkerSize', 6); hold on;
semilogy(0:nepoch, hist_vector-fstar, '-*', 'LineWidth', 1, 'MarkerSize', 3, 'Color',[41/255,41/255,255/255]); hold on;
hold off;
legend('Rand',  'Matrix',  'Vector', 'Location', 'best', 'FontSize', 12);
% legend('Rand Batch', 'Orth Batch' , 'Matrix Batch', 'Location', 'best', 'FontSize', 12);
%title('Convergence of Different Mini-batch Selection Methods', 'FontSize', 16);
xlabel('$k$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$f^k - f^*$ (Log Scale)', 'Interpreter', 'latex', 'FontSize', 14);
% grid on;
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.6);
set(gca, 'FontSize', 12);
box on;
%%  画 error——x
figure; 
semilogy(0:nepoch, hist_random_x, '-^', 'LineWidth', 1, 'MarkerSize',3,'Color', [5/255,5/255,5/255]); hold on;
% semilogy(0:nepoch, historth-fstar, '-s', 'LineWidth', 2, 'MarkerSize', 6);
semilogy(0:nepoch, hist_matrix_x, '-s', 'LineWidth', 1, 'MarkerSize', 3, 'Color', [255/255,17/255,14/255]);
% semilogy(0:nepoch, hist_well-fstar, '-*', 'LineWidth', 2, 'MarkerSize', 6); hold on;
semilogy(0:nepoch, hist_vector_x, '-*', 'LineWidth', 1, 'MarkerSize', 3, 'Color', [41/255,41/255,255/255]); hold on;
hold off;
legend('Rand',  'Matrix',  'Vector', 'Location', 'best', 'FontSize', 12);
% legend('Rand Batch', 'Orth Batch' , 'Matrix Batch', 'Location', 'best', 'FontSize', 12);
%title('Convergence of Different Mini-batch Selection Methods', 'FontSize', 16);
xlabel('$k$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\|x^k - x^*\|$ (Log Scale)', 'Interpreter', 'latex', 'FontSize', 14);
% grid on;
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.6);
set(gca, 'FontSize', 12);
box on;

%% write for bigbatch
dataFolderPath = '/home/ubuntu/user/xlj/opt/bigbatch/Datasets/MNIST_jpg/training/';
imds = imageDatastore(dataFolderPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% 显示图像数据存储信息
disp(imds);

% 初始化一个空的矩阵来存储抽样图像和路径列表
imageMatrix = [];
imagePaths = {}; % 存储选定图像的路径
targetSize = [28 28]; % 设置目标图像大小
datasize = 10000;
imds = shuffle(imds);
images = readall(imds);
k = 1;
% 将图像转换为目标大小的向量并添加到矩阵中
for i = (k-1)*datasize +1 : k *datasize
    grayImage = images{i}; % 已经是灰度图
    resizedImage = imresize(grayImage, targetSize); % 调整图像大小
    imageVector = reshape(resizedImage, 1, []); % 将图像展平为向量
    imageMatrix = [imageMatrix; imageVector]; % 添加到矩阵中
    % 记录图像路径
    imgfile = imds.Files(i);
    imagePaths{end + 1} = imgfile; % 将路径添加到列表中
end

% 将图像路径保存为 TXT 文件
fileID = fopen('imagePaths_10000.txt', 'w');
fprintf(fileID, '%s\n', imagePaths{:});
fclose(fileID);
disp('Image paths saved successfully');
%%
rng(1234);
x = [1 2 3 4 5 23 34 53 5];
y = shuffle(1:8);
