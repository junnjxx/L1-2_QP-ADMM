# 导入库
import os

# 文件路径
file_path = "/home/ubuntu/xlj/new_model/Datasets/MNIST_jpg/60000_mnist_256/mnist_sorted_matrix_60000_256.txt"
output_file_path = '/home/ubuntu/xlj/new_model/Datasets/MNIST_jpg/60000_mnist_256/mnist_sorted_matrix_60000_256.txt'

# 读取文件并添加标签
with open(file_path, 'r') as f:
    lines = f.readlines()

# 打开输出文件写入带标签的路径
with open(output_file_path, 'w') as f_out:
    for line in lines:
        line = line.strip()  # 移除行尾的换行符
        # 获取标签，即路径倒数第二级文件夹名称
        label = os.path.basename(os.path.dirname(line))
        # 写入路径和标签
        f_out.write(f"{line},{label}\n")

print(f"文件已成功保存为 {output_file_path}")