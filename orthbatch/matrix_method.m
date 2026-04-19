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
    iter = 1e7;
    beta = 10;  % |X-Y| %80:1000,0.01 400:1000,0.0001 1200:1000，0.0001
    eta =100000 %0.0001; % SUM(Y)
    

    %% 自适应
    beta =1000000000000; eta =10000;
    % % 试试固定参数
    % beta = 20;
    % eta = 0.01; 

    epsi = 1e-10;
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
        % X = rand(n_,k_);Y = rand(n_,k_);
        % Y = init_Y' + rand(1) * 1e-3 ;
        % X = Y + rand(size(Y))* 0.01;
        ori_Y = Y ;
        sa_obj = [];
        sa_error = [];
            
        for d = 1:n_iter
    
            % [X,Y, error_list,flag,Lambda_history,r_history,dxy_history,round_y_collect,y_non_ratio,lambda_fnorm,y_history,x_history,R_history] = splitadmm20240119(A,G,eta,beta,d,iter, X,Y,epsi);
            [X,Y, error_list,flag,Lambda_history,r_history,dxy_history,round_y_collect,y_non_ratio,lambda_fnorm] = splitadmm20240119(A,G,eta,beta,d,iter, X,Y,epsi);
        end
 
        % % 处理前两个变量的0值，避免semilogy绘图问题
        % r_processed = r_history;
        % dxy_processed = dxy_history;
        % 
        % % 替换0值为1e-16，避免对数坐标下的问题
        % r_processed(r_processed == 0) = 1e-20;
        %  dxy_processed(dxy_processed < 1e-20) = 1e-20;
        % 
        % 
        % % 创建一个大图，包含3个子图
        % % === 1. 设置图形尺寸（在绘图前！）===
        % fig = figure('Units', 'inches', 'Position', [100, 100, 800, 1000]); 
        % % 建议高度为 10 英寸（4 个子图需要更多垂直空间）
        % 
        % % 第一个子图：r_history 和 dxy_history
        % subplot(4, 1, 1); % 3行1列中的第1个
        % semilogy(r_processed, 'b-', 'LineWidth', 1.5);
        % hold on;
        % semilogy(dxy_processed, 'r--', 'LineWidth', 1.5);
        % legend('Residual Frobenius Norm (h)', 'X-Y Frobenius Norm(p)', 'Location', 'best');
        % title('Convergence Behavior of Residual Frobenius Norm and ‖X-Y‖ Frobenius Norm');
        % xlabel('Iteration Steps');
        % ylabel('Log-Scaled Frobenius Norm Value');
        % % 调小x轴标签字体（设置为9号，适配学术图表简洁风格，与siamart_mmd_admm1023.pdf图表字体协调）
        % xlabel('Iteration Steps', 'FontSize', 9);
        % % 调小y轴标签字体
        % ylabel('Log-Scaled Frobenius Norm Value', 'FontSize', 9);
        % grid on;
        % ylim([1e-15, 1e2]);
        % hold off;
        % 
        % % 第二个子图：round_y_collect
        % % 对数坐标版本
        % subplot(4, 1, 2);
        % semilogy(round_y_collect, 'g-', 'LineWidth', 2);
        % title('Evolution of Rounding Error');
        % xlabel('Iteration Steps');
        % ylabel('Log-Scaled Frobenius Norm ‖X-round(X)‖');
        % % 调小x轴标签字体（设置为9号，适配学术图表简洁风格，与siamart_mmd_admm1023.pdf图表字体协调）
        % xlabel('Iteration Steps', 'FontSize', 9);
        % % 调小y轴标签字体
        % ylabel('Log-Scaled Frobenius Norm Value', 'FontSize', 9);
        % grid on;
        % % 非对数坐标版本
        % % subplot(4, 1, 2);
        % % plot(round_y_collect, 'g-', 'LineWidth', 1.2);
        % % title('Evolution of Rounding Error');
        % % xlabel('Iteration Steps');
        % % ylabel('Frobenius Norm ‖X-round(X)‖');
        % % grid on;
        % 
        % 
        % 
        % % 第三个子图：y_non_ratio
        % subplot(4, 1, 3);
        % plot(y_non_ratio, 'm-', 'LineWidth', 2);
        % title('Proportion of Non-Binary Elements in X');
        % xlabel('Iteration Steps');
        % ylabel('Fraction of Non-0/1 Elements');
        % % 调小x轴标签字体（设置为9号，适配学术图表简洁风格，与siamart_mmd_admm1023.pdf图表字体协调）
        % xlabel('Iteration Steps', 'FontSize', 9);
        % % 调小y轴标签字体
        % ylabel('Log-Scaled Frobenius Norm Value', 'FontSize', 9);
        % grid on;
        % 
        % 
        % 
        % % 第4个子图：lambda f norm
        % subplot(4, 1, 4);
        % plot(lambda_fnorm, 'c-', 'LineWidth', 2);
        % title('Evolution of $\Lambda$ Frobenius Norm ','Interpreter', 'latex');
        % xlabel('Iteration Steps');
        % ylabel('$\Lambda$ Frobenius Norm','Interpreter', 'latex');
        % % 调小x轴标签字体（设置为9号，适配学术图表简洁风格，与siamart_mmd_admm1023.pdf图表字体协调）
        % xlabel('Iteration Steps', 'FontSize', 9);
        % % 调小y轴标签字体
        % ylabel('Log-Scaled Frobenius Norm Value', 'FontSize', 9);
        % grid on;
        % 
        % % % 调整子图之间的间距
        % % hSgtitle(sprintf('Variable Evolution Summary at $\\beta=%.2f$, $\\eta=%.2f$', beta, eta), 'Interpreter', 'latex');
        % % 
        % % set (hSgtitle, 'Position', [400, 1550, 0, 0]); % 手动设置总标题坐标，y 越大越靠上
        % % 
        % 
        % % === 设置图形尺寸 ===
        % 
        % 
        % % === 设置纸张参数 ===
        % fig.PaperSize = [8.5, 11];
        % fig.PaperPosition = [0, 0, 800, 1000]; % 与图形尺寸一致
        % fig.PaperOrientation = 'portrait';
        % 
        % % === 导出为高质量 PDF ===
        % print(fig, '-dpdf', '-r300', 'beta20eta0.01.pdf');
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







