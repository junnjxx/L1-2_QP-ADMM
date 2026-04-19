function [hist,I] = matrixbatch(x,Q,G,stepsize,bs,nepoch,Qa,data_vector, well_I, K, N,opt_type,gamma,lamb)
init_Y = init_Y_well_batch(N, K, bs);
[n,m] = size(G);
ga = sum(G,2);
nb = m / bs;
hist = zeros(nepoch+1, 1); hist(1) = getobj(x,Qa,ga);
fprintf('now matrix\n')
[I,ori_Y]   =  matrix_method(data_vector, bs, init_Y);
iter_count = 0;
for epoch = 1: nepoch
    random_order = randperm(size(I, 2));
    I = I(:, random_order);
    for iter =1: nb
        iter_count = iter_count +1;
        i = I(:,iter);
        Grad = getgrad(x,Q(i),G(:,i),opt_type,lamb,n);
        if opt_type == 0
            grad = sum(Grad,2);
            x = x - stepsize * grad;
        end
        if opt_type == 1
            grad = sum(Grad,2);
            x = x - stepsize * grad;
        end
        v = 0 ; 
        if opt_type == 2
            grad = sum(Grad,2);
            v = gamma .* v + stepsize .* grad;
            x = x - v;
        end
        s = 0 ; v = 0; beta1 = 0.9; beta2 = 0.999 ;epsilon = 1e-6;
        if opt_type == 3
            grad = sum(Grad,2);
            s = beta1 * s + (1 - beta1) * grad;
            v = beta1 * v + (1 - beta2) * grad .* grad;
            s2 = s / (1 - beta1^iter_count);
            v2 = v / (1 - beta2^iter_count);
            x = x - stepsize * s2 ./ (sqrt(v2) + epsilon);
        end
    end
    hist(epoch+1) = getobj(x,Qa,ga);
     
end
end
%%
function Grad = getgrad(x,Q,G,grad_type,lambda,n)
[n,m] = size(G);
Grad = zeros(n,m);
for i =1:m
    Grad(:,i) = Q{i}*x;
end
Grad = Grad + G;
if grad_type ==1
    Grad = Grad + (lambda/n .* x);
end


end
function f = getobj(x,Q,g)
    f = x'*Q*x/2+g'*x;
end

function [M_ , ori_Y] = matrix_method(  data , bs, init_Y)

    [m,n] = size(data);  % m是样本数量，n是样本的维度
    imageMatrix = data;
    pairwiseNorms = pdist(imageMatrix, 'euclidean');
    distanceMatrix = squareform(pairwiseNorms);
    bandwidth = sum(sum(distanceMatrix))./(m^2 - m);
    phi0 = exp(-distanceMatrix./bandwidth);
    
    % 参数设置
    n_iter = 1;
    iter = 1e6;
    beta = 1e3;  % |X-Y|
    eta = 1e0; % SUM(Y)
    epsi = 1e-5;
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
        X = ones(n_,k_)/ k_ + (rand(n_,k_)-0.5) * (1/k_) * 1e-2;
        Y = ones(n_,k_)/ k_ + (rand(n_,k_)-0.5) * (1/k_) * 1e-2;
        % Y = init_Y' + rand(1) * 1e-3 ;
        % X = Y + rand(size(Y))* 0.01;
        ori_Y = Y ;
        sa_obj = [];
        sa_error = [];
            
        for i = 1:n_iter
    
            [X,Y, error_list,flag] = splitadmm20240119(A,G,eta,beta,i,iter, X,Y,epsi);
            
        end



    
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







