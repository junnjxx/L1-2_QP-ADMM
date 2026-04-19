function [I] = matrix_method(bs,data_vector)
fprintf('now matrix\n')
[I,ori_Y]   =  matrix_detail(data_vector, bs);
end
%%
function [M_ , ori_Y] = matrix_detail(  data , bs)

    [m,n] = size(data);  % m是样本数量，n是样本的维度
    imageMatrix = data;
    pairwiseNorms = pdist(imageMatrix, 'euclidean');
    distanceMatrix = squareform(pairwiseNorms);
    bandwidth = sum(sum(distanceMatrix))./(m^2 - m);
    phi0 = exp(-distanceMatrix./bandwidth);
    
    % 参数设置
    n_iter = 1;
    iter = 1e6;
    beta = 5000;  % |X-Y|
    eta = 1; % SUM(Y)
    epsi = 1e-3;
    repeat =1;
    nb = floor(m / bs);
    k = m;
    m = nb;
    for i = 1:repeat
        G = phi0 * ones(k,m);
        %   根据一定的规则增大beta和eta
        A = phi0(1:(m*bs), 1:(m*bs)) ;
        A = A./bs./bs;
        G = G(1:(m*bs),:);
        G = -2 .* G ./bs ./k;
        [n_,k_] = size(G);
        % X = ones(n_,k_)/ k_ + (rand(n_,k_)-0.5) * (1/k_) * 1e-2;
        % Y = ones(n_,k_)/ k_ + (rand(n_,k_)-0.5) * (1/k_) * 1e-2;
        X = rand(n_,k_);
        Y = rand(n_,k_);
        Lambda = zeros(n_,k_);

        % X = readmatrix("/home/ubuntu/xlj/opt/bigbatch/cifar10_datasets/cifar10_matrix_resume/matrix_X_0_X_nnzY_50619_iter_30000.csv");
        % Y = readmatrix("/home/ubuntu/xlj/opt/bigbatch/cifar10_datasets/cifar10_matrix_resume/matrix_X_0_Y_nnzY_50619_iter_30000.csv");
        % Lambda = readmatrix("/home/ubuntu/xlj/opt/bigbatch/cifar10_datasets/cifar10_matrix_resume/matrix_X_0_LAMBDA_nnzY_50619_iter_30000.csv");
        % Y = init_Y' + rand(1) * 1e-3 ;
        % X = Y + rand(size(Y))* 0.01;
        ori_Y = Y ;

        for d = 1:n_iter
    
            [X,Y, error_list,flag] = splitadmm20240119(A,G,eta,beta,d,iter, X,Y,Lambda,epsi);
            
        end
        writematrix(Y,'cifar10_matrix_Y_50000_250_HOG.csv');



    
    end

    
    M = [];
    for i = 1:m
        m_temp = find(any(Y(:,i), 2))';
        M = vertcat(M,m_temp );
    end
    M_ = M';
    



end

function init_Y = init_Y_well_batch(n, K, batch_size)
    N = n * K ; 
    data = 1:N;
    n_n = floor(batch_size / K);
    nb = floor(N/batch_size);
    % 按每 n 个数字分成一组
    classes = mat2cell(data, 1, repmat(n, 1, K));

    % 均匀地从每组中选择 batch_size 个样本
    batches = [];
    init_Y = rand(N,nb) * 0.2;
    for i = 1:nb
        batch = [];
        for j = 1:K
            cls = classes{j};
            % 从每组中选择适当数量的样本
            batch = [batch, cls((i-1)*n_n + 1:i*n_n)];
        end
        
        % 保证 batch 的大小为 batch_size
        if length(batch) > batch_size
            batch = batch(1:batch_size);  
        end
        init_Y(batch,i) = rand(batch_size,1) * 0.1 + 0.7;
        batches = [batches; batch];
    end
    
    % 转换为矩阵
    batches = reshape(batches , [], batch_size);
    
   

end







