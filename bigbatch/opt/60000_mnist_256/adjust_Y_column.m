% 读取 CSV 文件为矩阵 Y
bs =256 ;
csv_file = 'mnist_matrix_60000_1.csv';  % 替换为实际的文件路径
Y = readmatrix(csv_file);    % 加载为矩阵

% Step 1: 将列和为 256 的列移动到矩阵前面
col_sums = sum(Y, 1);                % 计算每列的列和
cols_equal_bs = find(col_sums == bs);  % 找到列和为 256 的列索引
cols_not_bs = find(col_sums ~= bs);   % 找到列和不等于 256 的列索引

% 重排矩阵
Y = [Y(:, cols_equal_bs), Y(:, cols_not_bs)];

% Step 2: 对后面的列进行操作
col_sums = sum(Y, 1);                % 更新列和
start_idx = length(cols_equal_bs) + 1;  % 后面需要处理的列的起始索引

% 获取列和不足 256 和超过 256 的列索引
cols_underfilled = find(col_sums < bs & (1:size(Y, 2)) >= start_idx);
cols_overfilled = find(col_sums > bs & (1:size(Y, 2)) >= start_idx);

% 补充列和不足 256 的列
for i = 1:length(cols_underfilled)
    under_idx = cols_underfilled(i);
    deficit = bs - col_sums(under_idx);
    
    for j = 1:length(cols_overfilled)
        over_idx = cols_overfilled(j);
        if col_sums(over_idx) <= bs
            continue;
        end
        surplus = col_sums(over_idx) - bs;
        transfer = min(deficit, surplus);
        
        % 找到列中值为 1 的位置
        ones_indices = find(Y(:, over_idx) == 1);
        
        % 随机选择需要转移的行索引
        transfer_indices = ones_indices(randperm(length(ones_indices), transfer));
        
        % 转移值
        Y(transfer_indices, over_idx) = 0;
        Y(transfer_indices, under_idx) = 1;
        
        % 更新列和和缺口
        col_sums(under_idx) = col_sums(under_idx) + transfer;
        col_sums(over_idx) = col_sums(over_idx) - transfer;
        deficit = deficit - transfer;
        
        if deficit == 0
            break;
        end
    end
end

% Step 3: 保存调整后的矩阵
output_csv_file = 'adjusted_Y_mnist_matrix.csv';  % 替换为实际的输出文件路径
writematrix(Y, output_csv_file);

disp(['调整后的矩阵已保存到 ', output_csv_file]);

% 获取每行中值为 1 的列索引
[row_indices,~] = find(Y);

%%
csv_file = 'save_B.csv';  % 替换为实际的文件路径
row_indices = readmatrix(csv_file); 
row_indices = int32(row_indices+1);
% 读取图片路径文件
fileID = fopen('train.txt', 'r');
X = textscan(fileID, '%s'); % 读取为字符串列表
fclose(fileID);

% 转换为元胞数组
X = X{1}; % 提取路径列表


% 确保索引的合法性
n_sort = 59904; % 只对前 30000 条路径进行排序
if length(row_indices) < n_sort
    error('索引数量不足以覆盖需要排序的路径数量。');
end

if n_sort > length(X)
    error('X 文件中的路径数量少于 59904，请检查输入文件。');
end

% 分离需要排序和未排序部分
X_to_sort = X(1:n_sort); % 提取前 30000 条路径
X_unsorted = X(n_sort+1:end); % 保留剩余路径

% 对 X_to_sort 按索引排序
X_sorted = X_to_sort(row_indices);

% 将排序后的路径与未排序部分合并
X_final = [X_sorted; X_unsorted];

% 保存到新的文件
fileID = fopen('mnist_sorted_vector_60000_256.txt', 'w');
fprintf(fileID, '%s\n', X_final{:});
fclose(fileID);

disp('路径排序完成，已保存至 mnist_sorted_vector_60000_256.txt');


