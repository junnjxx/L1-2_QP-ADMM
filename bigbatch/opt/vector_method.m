function save_B = vector_method(bs,phi0)

[~ , n] = size(phi0);
nb = floor(n / bs );
D = 1: nb*bs ;
save_B = [];
Q = D;
B = [];
P = [];

for i = 1:(nb-1)
    P = sort(vertcat(P, Q(B)));
    Q =  sort(setdiff(Q, Q(B)));
    np = size(P(:),1);
    nq = size(Q,2);
    phi1 = phi0(Q,Q);
    phi2 = sum(phi1).*((np+bs)/n);
    phi3 = sum(phi0(Q,P)').*((nq-bs)/n);
    
    % 创建 Z
    for j = 1:nq
        if np >0
            phi1(j,j) = phi1(j,j) - phi2(j) + phi3(j);
        else
            phi1(j,j) = phi1(j,j) - phi2(j);
        end
    end
   
    n_iter = 1;
    iter = 2e5;
    % X = bs* ones(nq,1)/nq   + ((rand(nq,1)-0.5)/ nq * 1e-1);
    % Y = bs * ones(nq,1)/nq  + ((rand(nq,1)-0.5)/ nq * 1e-1);
    % X = rand(nq,1); Y = rand(nq,1);
    eta = 10000;
    beta = 1000000 ;
    logFile = 'nnz_log_cifar10_50000_250.txt';
    fileID = fopen(logFile, 'a'); % 'a' 表示追加模式
    for j = 1:n_iter
        %  X = bs* ones(nq,1)/nq   + ((rand(nq,1)-0.5)/ nq * 1e-1);
        %  Y = bs * ones(nq,1)/nq  + ((rand(nq,1)-0.5)/ nq * 1e-1);
         X = rand(nq,1); Y = rand(nq,1);
        [~,Y] = splitadmm20240307(phi1,nq, bs,eta,beta,iter,X,Y );  
        if nnz(Y) == bs
            break
        else
            if nnz(Y) < bs
                zeroIndices = find(Y == 0);
                selectedIndices = randsample(zeroIndices, bs-nnz(Y));
                fprintf(fileID, 'NNZ (Y) < BS: Added %d ones to Y. Updated NNZ(Y) = %d\n', ...
        (bs - nnz(Y)), nnz(Y));
                Y(selectedIndices) = 1;
                % 记录到日志文件
                
            else
                oneIndices = find(Y ==1);
                selectedIndices = randsample(oneIndices, nnz(Y)-bs);
                fprintf(fileID, 'NNZ (Y) > BS: Removed %d ones from Y. Updated NNZ(Y) = %d\n', ...
        (nnz(Y) - bs), nnz(Y));
                Y(selectedIndices) = 0;
                
            end
        end
    end
    B = find(Y==1);
    s_B = Q(B);
    save_B = [save_B;s_B];

end
    left_Q = sort(setdiff(Q, s_B));
    save_B =[save_B;left_Q];
    save_B = save_B';

end
