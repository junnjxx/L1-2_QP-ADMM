function save_B = vector_method(bs,data_vector)
imageMatrix = data_vector;
n = size(imageMatrix,1);
pairwiseNorms = pdist(imageMatrix, 'euclidean');
distanceMatrix = squareform(pairwiseNorms);
%% 计算这个矩阵的均值/ 去除对角线 / 计算bandwidth / 计算 phi0
% bandwidth = sum(sum(distanceMatrix))./(n^2 - n);
bandwidth =1;
% 计算原始的 大phi矩阵
phi0 = exp(-distanceMatrix./bandwidth);
%% define phi1 phi phi3
D = 1:n;  % 所有数据的索引
B = [];
P = [];
Q = D;
selected = [];
saved_q = [];
for i = 1:(n/bs)-1
    P = sort(vertcat(P, Q(B)));
    Q  = sort(setdiff(Q, Q(B)));
    np = size(P(:),1);
    nq = size(Q,2);
    phi1 = phi0(Q,Q);
    phi2 = sum(phi0(Q,Q),2).*((np+bs)/n);
    phi3 = sum(phi0(Q,P(:)),2).*((nq-bs)/n);
    Z = phi1;
    % 创建 Z
    for j = 1:nq
        if np >0
            Z(j,j) = Z(j,j) - phi2(j) + phi3(j);
        else
            Z(j,j) = Z(j,j) - phi2(j);
        end
    end
    % 进行优化
%     syms x [1 n+n*n]
    ZT = Z'
    f = [zeros(1,nq),ZT( : )'];
    Aeq = [ones(1, nq), zeros(1, nq*nq)];
    beq = bs;
    A = zeros(nq*nq,nq+nq*nq);
    b = zeros(nq*nq,1);
    z_flatten = ZT(:);
    for t = 1:length(z_flatten)
        a = zeros(1,nq+nq*nq);
        i = ceil(t / nq);
        j = mod(t-1, nq) + 1;
        if Z(i,j) <= 0
            if i==j
                a(i) = -2;a(nq+t) = 2;
            else
                a(i) = -1; a(j)= -1; a(nq+t) = 2;
            end
            b(t) = 0 ;
        else
            if i==j
                a(i) = 2;a(nq+t) = -2;
            else
                a(i) = 1; a(j)= 1; a(nq+t) = -2;
            end
            b(t) = 1;
        end
        A(t,:) = a;

    end
    lb = zeros(1,nq+nq*nq);
    f_double = double(f)
    ub = ones(1,nq+nq*nq);
    options = optimoptions('linprog','Algorithm','interior-point');
    [new_x, fval] = linprog(f_double,A,b, Aeq, beq,lb,ub, options)
    % [new_x, fval] = linprog(f_double,A,b, Aeq, beq,lb,ub)
    [~, sortedIndices] = sort(new_x(1:nq), 'descend');
    B = sortedIndices(1:bs);
    selected =  [selected ; Q(B)];
end
left =  sort(setdiff(Q, Q(B)));
selected =  [selected ; left]

end 