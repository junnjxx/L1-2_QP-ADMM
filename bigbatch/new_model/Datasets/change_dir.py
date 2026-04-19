# 定义旧路径和新路径
old_path = '/home/ubuntu/xlj/opt/bigbatch/Datasets/MNIST_jpg/training'
new_path = '/home/ubuntu/xlj/new_model/Datasets/MNIST_jpg/training'

# 定义文件路径
file_path = '/home/ubuntu/xlj/new_model/Datasets/MNIST_jpg/60000_mnist_256/mnist_sorted_matrix_60000_256.txt'

# 读取文件并替换路径
with open(file_path, 'r') as file:
    lines = file.readlines()

# 替换路径
updated_lines = [line.replace(old_path, new_path) for line in lines]

# 将修改后的内容写回文件
with open(file_path, 'w') as file:
    file.writelines(updated_lines)

print("文件中的路径已成功更新！")