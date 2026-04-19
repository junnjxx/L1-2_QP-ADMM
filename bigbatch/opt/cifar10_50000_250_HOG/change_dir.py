
old_path = '/home/ubuntu/xlj/opt/bigbatch/cifar10_datasets/train_cifar10_gray/'
new_path = '/home/ubuntu/xlj/opt/bigbatch/cifar10_datasets/train_cifar10/'


file_path = "/home/ubuntu/xlj/opt/bigbatch/cifar10_50000_250_HOG/cifar10_sorted_vector_50000_250_HOG.txt"


with open(file_path, 'r') as file:
    lines = file.readlines()


updated_lines = [line.replace(old_path, new_path) for line in lines]


with open(file_path, 'w') as file:
    file.writelines(updated_lines)

print("ok")