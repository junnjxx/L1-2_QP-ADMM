function [hist, I] = wellbatch(x,Q,G,stepsize,bs,nepoch,Qa,K,N,opt_type,gamma,lamb)

[n,m] = size(G);
ga = sum(G,2);
nb = m / bs;
hist = zeros(nepoch+1, 1); hist(1) = getobj(x,Qa,ga);
I  =  well_method(N, K, bs);
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

function batches = well_method(n, K, batch_size)
    N = n * K ; 
    data = 1:N;
    n_n = floor(batch_size / K);

    % 按每 n 个数字分成一组
    classes = mat2cell(data, 1, repmat(n, 1, K));

    % 均匀地从每组中选择 batch_size 个样本
    batches = [];
    for i = 1:floor(N / batch_size)
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
        
        batches = [batches; batch];
    end
    
    % 转换为矩阵
    batches = reshape(batches , [], batch_size);
    batches = batches';
end








