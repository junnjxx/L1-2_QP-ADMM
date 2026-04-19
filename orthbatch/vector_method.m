function save_B = vector_method(bs,data_vector)
imageMatrix = data_vector;
n = size(imageMatrix,1);
pairwiseNorms = pdist(imageMatrix, 'euclidean');
distanceMatrix = squareform(pairwiseNorms);
%% 计算这个矩阵的均值/ 去除对角线 / 计算bandwidth / 计算 phi0
bandwidth = sum(sum(distanceMatrix))./(n^2 - n);
% 计算原始的 大phi矩阵
phi0 = exp(-distanceMatrix./bandwidth);
[n , n] = size(phi0);
nb = floor(n / bs );
repeat =1;
for d = 1: repeat
D = 1: nb*bs ;
save_B = [];
Q = D;
B = [];
P = [];

for i = 1:(nb)
    fprintf('now is batch %d',i);
    P = sort(vertcat(P, Q(B)));
    Q =  sort(setdiff(Q, Q(B)));
    np = size(P(:),1);
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
   
    n_iter = 1;
    iter = 2e5;
    X = bs* ones(nq,1)/nq   + ((rand(nq,1)-0.5)/ nq * 1e-1);
    Y = bs * ones(nq,1)/nq  + ((rand(nq,1)-0.5)/ nq * 1e-1);
    % X = rand(nq,1); Y = rand(nq,1);
    eta = 0.001;
    beta = 100 ;
  
    for j = 1:n_iter
        [X,Y] = splitadmm20240307(Z,nq, bs,eta,beta,iter,X,Y );   
    end
    B = find(Y==1);
    s_B = Q(B);
    save_B = [save_B;s_B];

end
    left_Q = sort(setdiff(Q, s_B));
    save_B =[save_B;left_Q];
    save_B = save_B';
   
end

end
