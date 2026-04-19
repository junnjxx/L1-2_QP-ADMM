import json
import random

# 加载 JSON 文件
json_dir = "/home/ubuntu/xlj/opt/bigbatch/cifar10_datasets/annotations/cifar10_train.json"
save_dir = "/home/ubuntu/xlj/opt/bigbatch/cifar10_datasets/train_gray.txt"
# 数据路径前缀
prefix = "/home/ubuntu/xlj/opt/bigbatch/cifar10_datasets/train_cifar10_gray/"
with open(json_dir, "r") as json_file:
    annotations = json.load(json_file)

# 确保 images 和 categories 长度一致
assert len(annotations["images"]) == len(annotations["categories"]), "Images and categories must have the same length."


# 组合图片路径和标签，并添加前缀
data = [(f"{prefix}{image}", label) for image, label in zip(annotations["images"], annotations["categories"])]

# 随机打乱数据顺序
# random.shuffle(data)

# 写入到 annotations.txt 文件
with open(save_dir, "w") as txt_file:
    for image_path, label in data:
        txt_file.write(f"{image_path},{label}\n")

print("转换完成生成文件")