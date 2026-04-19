% 假设三个 CSV 文件的文件名分别为 'loss1.csv', 'loss2.csv', 'loss3.csv'

% 读取 CSV 文件
data1 = readtable('/home/ubuntu/xlj/new_model/results/mnist_59904_256/sgd/run-train_loss_Random-tag-train_loss (1).csv');  % 读取第一个 CSV 文件
data2 = readtable('/home/ubuntu/xlj/new_model/results/mnist_59904_256/sgd/run-train_loss_Matrix-tag-train_loss.csv');  % 读取第二个 CSV 文件
data3 = readtable('/home/ubuntu/xlj/new_model/results/mnist_59904_256/sgd/run-train_loss_Vector-tag-train_loss.csv');  % 读取第三个 CSV 文件

% 假设 CSV 文件中有 'epoch' 和 'trainloss' 列
epoch1 = data1.Step;  % 第一个文件的 epoch 列
trainloss1 = data1.Value;  % 第一个文件的 trainloss 列

epoch2 = data2.Step;  % 第二个文件的 epoch 列
trainloss2 = data2.Value;  % 第二个文件的 trainloss 列

epoch3 = data3.Step;  % 第三个文件的 epoch 列
trainloss3 = data3.Value;  % 第三个文件的 trainloss 列

% 如果存在非正数，进行处理
trainloss1(trainloss1 <= 0) = NaN;  % 将非正值替换为 NaN
trainloss2(trainloss2 <= 0) = NaN;
trainloss3(trainloss3 <= 0) = NaN;

% 绘制图形
figure;  % 新建一个图形窗口
hold on;  % 保持当前图形，允许多个图形绘制

semilogy(1:100,trainloss1, '-^', 'LineWidth', 1, 'MarkerSize',3,'Color', [5/255,5/255,5/255]); hold on;
% semilogy(0:nepoch, historthq a-fstar, '-s', 'LineWidth', 2, 'MarkerSize', 6);
semilogy(1:100, trainloss2, '-s', 'LineWidth', 1, 'MarkerSize', 3, 'Color', [41/255,41/255,255/255]);
% semilogy(0:nepoch, hist_well-fstar, '-*', 'LineWidth', 2, 'MarkerSize', 6); hold on;
semilogy(1:100, trainloss3, '-*', 'LineWidth', 1, 'MarkerSize', 3, 'Color', [255/255,17/255,14/255]); hold on;
hold off;
legend('Rand',  'Matrix',  'Vector', 'Location', 'best', 'FontSize', 12);

xlabel('Epoch', 'FontSize', 14);
ylabel('Train Loss (Log Scale)', 'FontSize', 14);

hold off;  % 结束绘图