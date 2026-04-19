import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# 读取CSV文件的函数

def readcsv(files_dir, metric):
    # 读取CSV文件
    plots = pd.read_csv(
        files_dir,
        dtype=np.float64,  # 核心：用64位浮点数（默认可能是float32，精度低）
        na_filter=False    # 避免空值填充影响精度
    )
    
    # 假定数据在第三列 (index 2) 和第四列 (index 3)，根据实际文件结构进行修改
    x = plots.iloc[:, 1].values  # 第三列 (Steps)
    y = plots.iloc[:, 2].values  # 第四列 (Score)
    if metric == 'test_ac':
        y = 1 - y  # 如果是 test_ac，则计算错误率（1 - accuracy）
    # y = [round(val, 4) for val in y]
    return x, y

# 设置字体
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

# 选择要绘制的 metric
metric = 'test_ac'  # 也可以是 'train_loss' 或 'test_loss' test_ac
big_flag = 0
basic_data_dir = '/Users/jun/research/RES/code/filed_20251115/xlj_236/new_model/results/mnist_59904_256/sgd/'
# basic_data_dir = '/Users/jun/research/RES/code/filed_20251115/xlj_236/new_model/results/cifar10_50000_250_HOG/sgd/'
# 根据 metric 设置文件路径和 y 轴标签
if metric == 'train_loss':
    output_path = os.path.join(basic_data_dir,  "cifar10_train_loss_plot_big_new.png")  
    output_path_eps = os.path.join(basic_data_dir, "cifar10_train_loss_plot_big_new.eps")
    x1, y1 = readcsv(os.path.join(basic_data_dir,"run-train_loss_Random-tag-train_loss.csv"), metric)
    x2, y2 = readcsv(os.path.join(basic_data_dir,"run-train_loss_Matrix-tag-train_loss.csv"), metric)
    x3, y3 = readcsv(os.path.join(basic_data_dir,"run-train_loss_Vector-tag-train_loss.csv"), metric)
    y_label = 'Train Loss (Log  Scale)'
elif metric == 'test_loss':
    output_path = os.path.join(basic_data_dir, "mnist_test_loss_plot.png")
    output_path_eps = os.path.join(basic_data_dir,"mnist_test_loss_plot.eps")
    x1, y1 = readcsv(os.path.join(basic_data_dir,"run-test_loss_Random-tag-test_loss.csv"), metric)
    x2, y2 = readcsv(os.path.join(basic_data_dir,"run-test_loss_Matrix-tag-test_loss.csv"), metric)
    x3, y3 = readcsv(os.path.join(basic_data_dir,"run-test_loss_Vector-tag-test_loss.csv"), metric)
    y_label = 'Test Loss (Log  Scale)'
elif metric == 'test_ac':
    output_path = os.path.join(basic_data_dir,"mnist_test_error_plot_big_new.png")
    output_path_eps = os.path.join(basic_data_dir,"mnist_test_error_plot_big_new.eps")
    x1, y1 = readcsv(os.path.join(basic_data_dir,"run-test_accuracy_Random-tag-test_accuracy.csv"), metric)
    x2, y2 = readcsv(os.path.join(basic_data_dir,"run-test_accuracy_Matrix-tag-test_accuracy.csv"), metric)
    x3, y3 = readcsv(os.path.join(basic_data_dir,"run-test_accuracy_Vector-tag-test_accuracy.csv"), metric)
    y_label = 'Test Error (Log  Scale)'



# 绘制图形
# plt.figure(figsize=(10, 6))
# plt.plot(x1, y1, label='Random', color='black',linestyle='-')
# plt.plot(x2, y2, label='Matrix', color='red',linestyle='-.')
# plt.plot(x3, y3, label='Vector', color='blue',linestyle='--')
# # 设置坐标轴
# plt.yscale('log')  # 对y轴使用对数刻度
# plt.xlabel('Epochs')
# plt.ylabel(y_label)
# plt.legend(loc='upper right')
# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# # 绘制主图的曲线
plt.plot(x1, y1, label='Random', color='black',linestyle='-')
plt.plot(x2, y2, label='Matrix', color='red',linestyle='-.')
plt.plot(x3, y3, label='Vector', color='blue',linestyle='--')

ax.set_yscale('log')  # 将 y 轴设置为对数
ax.set_xlim(0, 100)
ax.set_xlabel('Epochs', fontsize=10)
ax.set_ylabel(y_label, fontsize=10)
ax.legend(fontsize=10)
if big_flag ==1:
    # 添加嵌入式图形（图中图）
    # loc="center" 和 bbox_to_anchor=(0.7, 0.5) 表示将嵌入式图形放在图的中间偏右的位置
    ax = plt.gca()  # 获取当前的轴
    axins = inset_axes(ax, width="30%", height="40%", loc="right" )

    # 在图中图中绘制数据，放大某个区域
    axins.plot(x1, y1, color='black',linestyle='-')
    axins.plot(x2, y2, color='red',linestyle='-.')
    axins.plot(x3, y3, color='blue',linestyle='--')
    axins.set_xlim(85, 98)  # 设置放大的 x 轴范围
    axins.set_ylim(0.00015, 0.00025)  # 设置放大的 y 轴范围 sgdwm
    # axins.set_ylim(0.0035, 0.0055)
    # 在主图中添加矩形框，显示被放大的区域
    # from matplotlib.patches import Rectangle
    # rect = Rectangle((10, 0.1), 10, 0.9, linewidth=1, edgecolor='red', facecolor='none')
    # ax.add_patch(rect)

    # 将主图和 inset 图进行关联
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")

# # 显示并保存图形
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
plt.savefig(output_path, dpi=300)
plt.savefig(output_path_eps, format='eps', dpi=300)

print(f"图像已保存到: {output_path}")