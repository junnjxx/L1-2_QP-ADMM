% 读取 CSV 文件为表格
matrix = readtable("/home/ubuntu/xlj/opt/bigbatch/datasets/cifar10_60000_10000/cifar10_vector_60000_10000_1.csv");
matrix = table2array(matrix); % 转换表格为矩阵

% 打开 TXT 文件
fileID = fopen("/home/ubuntu/xlj/opt/bigbatch/datasets/cifar10_60000_10000/CIFAR10_imagePaths_60000_10000_1.txt", 'r');
outputFile = '/home/ubuntu/user/xlj/opt/bigbatch/datasets/cifar10_60000_10000/cifar10_imagePaths_sorted_vector_60000_10000_1.txt';

% 初始化存储行内容的元胞数组
lines = {};

% 按行读取文件
line = fgetl(fileID); % 读取第一行
while ischar(line) % 检查是否为字符行
    lines{end+1} = line; % 将当前行添加到元胞数组
    line = fgetl(fileID); % 读取下一行
end

% 关闭文件
fclose(fileID);

% 确保 matrix 的值是有效索引
if all(matrix > 0 & matrix <= numel(lines))
    % 提取对应的行
    selectedLines = lines(matrix(:));
    
    % 保存为新的 TXT 文件
    
    fileID = fopen(outputFile, 'w'); % 以写入模式打开
    for i = 1:numel(selectedLines)
        fprintf(fileID, '%s\n', selectedLines{i});
    end
    fclose(fileID);
    
    disp(['选定的行内容已保存到: ', outputFile]);
else
    disp('Error: matrix 包含无效索引！');
end