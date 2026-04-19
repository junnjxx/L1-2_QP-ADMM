import os

# MNIST 图像数据的根目录
root_dir = '/home/ubuntu/xlj/new_model/Datasets/MNIST_jpg/testing'  # 替换为你自己的路径
output_file = '/home/ubuntu/xlj/new_model/Datasets/MNIST_jpg/testing.txt'  # 输出txt文件路径

# 打开txt文件用于写入
with open(output_file, 'w') as file:
    # 遍历0-9文件夹，每个文件夹代表一个标签
    for label in range(10):
        folder_path = os.path.join(root_dir, str(label))
        
        # 确保文件夹存在
        if os.path.exists(folder_path):
            # 获取并排序文件夹中的所有图片文件
            for img_file in sorted(os.listdir(folder_path)):
                # 生成每个图片的完整路径
                img_path = os.path.join(folder_path, img_file)
                
                # 确保是文件而不是目录
                if os.path.isfile(img_path):
                    # 写入格式：图像路径 标签
                    file.write(f"{img_path},{label}\n")
        else:
            print(f"Folder {folder_path} does not exist.")