from torch.utils.data import Dataset
from PIL import Image
import cv2
from torchvision import transforms
import numpy as np
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import random
def csv_to_Matrix(path):
    x_Matrix = pd.read_csv(path, header=None)
    x_Matrix = np.array(x_Matrix)
    return x_Matrix


class MyDataset_mnist(Dataset):
    def __init__(self, file_path,data_size,S,bs):
        with open(file_path, 'r') as file:
            self.namelist = file.readlines()
            self.namelist = self.namelist[:data_size]
        # Shuffle based on S mode
        if S in ["Matrix", "Vector"]:
            self.namelist = self.shuffle_groups(self.namelist, bs)
        self.frame_list = []
        for name in self.namelist:
            data_info = name.strip().split(',')
            self.frame_list.append([data_info[0], int(data_info[1])])
        print(len(self.frame_list))
        self.lengt = len(self.frame_list)
    
            # self.preprocessing = transforms.Compose([
            #     transforms.Resize((32, 32)), # Resize to fit ResNet input
            #     transforms.ToTensor(),
            # ])
        self.preprocessing = transforms.Compose([
            transforms.Resize((32, 32)),  # 调整到 32x32 以适配 ResNet
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # MNIST 训练集的均值和标准差
])

    @staticmethod
    def shuffle_groups(namelist, group_size):
        groups = [namelist[i:i + group_size] for i in range(0, len(namelist), group_size)]
        for group in groups:
            random.shuffle(group)
        random.shuffle(groups)
        return [item for group in groups for item in group]
    def getframe(self, index):
        info = self.frame_list[index]
        data_path = info[0]
        data_label = info[1]
        # x_data = Image.open(data_path).convert('L') 
        x_data = Image.open(data_path)
        
        y_data =  data_label
        return x_data, y_data

    def __getitem__(self, index):
        x_data, y_data = self.getframe(index)
        x_data_tensor = self.preprocessing(x_data).float()
        # x_data_tensor = x_data_tensor.view(-1) # 对mlp模型需要，其他的需要注释掉
        y_data_tensor = torch.tensor(y_data)
        
        return x_data_tensor, y_data_tensor
    
    def __len__(self):
        return self.lengt

class MyDataset_mnist_test(Dataset):
    def __init__(self, file_path,data_size):
        with open(file_path, 'r') as file:
            self.namelist = file.readlines()
            self.namelist = self.namelist[:data_size]

        self.frame_list = []
        for name in self.namelist:
            data_info = name.strip().split(',')
            self.frame_list.append([data_info[0], int(data_info[1])])
        print(len(self.frame_list))
        self.lengt = len(self.frame_list)
    
        self.preprocessing = transforms.Compose([
            transforms.Resize((32, 32)), # Resize to fit ResNet input
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # MNIST 训练集的均值和标准差
        ])
    def getframe(self, index):
        info = self.frame_list[index]
        data_path = info[0]
        data_label = info[1]
        # x_data = Image.open(data_path).convert('L') 
        x_data = Image.open(data_path)
        
        y_data =  data_label
        return x_data, y_data

    def __getitem__(self, index):
        x_data, y_data = self.getframe(index)
        x_data_tensor = self.preprocessing(x_data).float()
        # x_data_tensor = x_data_tensor.view(-1) # 对mlp模型需要，其他的需要注释掉
        y_data_tensor = torch.tensor(y_data)
        
        return x_data_tensor, y_data_tensor
    
    def __len__(self):
        return self.lengt
    
class MyDataset_cifar10(Dataset):
    def __init__(self, file_path,data_size,S,bs):
        with open(file_path, 'r') as file:
            self.namelist = file.readlines()
        # Shuffle based on S mode
        if S in ["Matrix", "Vector"]:
            self.namelist = self.shuffle_groups(self.namelist, bs)
        self.frame_list = []
        for name in self.namelist:
            data_info = name.strip().split(',')
            self.frame_list.append([data_info[0], int(data_info[1])])
        self.frame_list = self.frame_list[:data_size]
        print(len(self.frame_list))
        self.lengt = len(self.frame_list)
    
        # self.preprocessing = transforms.Compose([
        #     transforms.Resize((32, 32)), # Resize to fit ResNet input
        #     transforms.ToTensor(),
        # ])
        self.preprocessing = transforms.Compose([
                 transforms.ToTensor()
            , transforms.RandomCrop(32, padding=4)  # 先四周填充0，在吧图像随机裁剪成32*32
            , transforms.RandomHorizontalFlip(p=0.5)  # 随机水平翻转 选择一个概率概率
            , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
            ])
    @staticmethod
    def shuffle_groups(namelist, group_size):
        groups = [namelist[i:i + group_size] for i in range(0, len(namelist), group_size)]
        for group in groups:
            random.shuffle(group)
        random.shuffle(groups)
        return [item for group in groups for item in group]
    def getframe(self, index):
        info = self.frame_list[index]
        data_path = info[0]
        data_label = info[1]
        # x_data = Image.open(data_path).convert('L') 
        x_data = Image.open(data_path)
        
        y_data =  data_label
        return x_data, y_data

    def __getitem__(self, index):
        x_data, y_data = self.getframe(index)
        x_data_tensor = self.preprocessing(x_data).float()
        # x_data_tensor = x_data_tensor.view(-1) # 对mlp模型需要，其他的需要注释掉
        y_data_tensor = torch.tensor(y_data)
        
        return x_data_tensor, y_data_tensor
    
    def __len__(self):
        return self.lengt
    
class MyDataset_cifar10_test(Dataset):
    def __init__(self, file_path,data_size):
        with open(file_path, 'r') as file:
            self.namelist = file.readlines()
        self.frame_list = []
        for name in self.namelist:
            data_info = name.strip().split(',')
            self.frame_list.append([data_info[0], int(data_info[1])])
        self.frame_list = self.frame_list[:data_size]
        print(len(self.frame_list))
        self.lengt = len(self.frame_list)

        self.preprocessing = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
            ])
    def getframe(self, index):
        info = self.frame_list[index]
        data_path = info[0]
        data_label = info[1]
        # x_data = Image.open(data_path).convert('L') 
        x_data = Image.open(data_path)
        
        y_data =  data_label
        return x_data, y_data

    def __getitem__(self, index):
        x_data, y_data = self.getframe(index)
        x_data_tensor = self.preprocessing(x_data).float()
        # x_data_tensor = x_data_tensor.view(-1) # 对mlp模型需要，其他的需要注释掉
        y_data_tensor = torch.tensor(y_data)
        
        return x_data_tensor, y_data_tensor
    
    def __len__(self):
        return self.lengt
class MyDataset_sy_data(Dataset):
    def __init__(self, data_path, sample_order_path):
        data = csv_to_Matrix(data_path)
        sample_order = csv_to_Matrix(sample_order_path)
        self.data = data[sample_order.flatten(), :]  # 使用 flatten() 将 sample_order 转换为一维数组
        # np.savetxt('./sy_simulation/sampled_data.csv', self.data, delimiter=',')
        self.length = self.data.shape[0]  # 使用 shape[0] 获取数组的行数
        print(self.length)

    def getframe(self, index):
        frame_data = self.data[index].reshape(1,-1)
        x_data = frame_data[:,:-1]
        y_data = frame_data[:,-1].reshape(1, 1)
        return x_data, y_data

    def __getitem__(self, index):
        x_data, y_data = self.getframe(index)
        x_data_tensor = torch.tensor(x_data)
        y_data_tensor = torch.tensor(y_data)
        return x_data_tensor, y_data_tensor

    def __len__(self):
        return self.length


class MyDataset_sy_data_test(Dataset):
    def __init__(self, data_path):
        data = csv_to_Matrix(data_path)
        self.length = data.shape[0]  # 使用 shape[0] 获取数组的行数
        print(self.length)
        self.data = data

    def getframe(self, index):
        frame_data = self.data[index].reshape(1,-1)
        x_data = frame_data[:,:-1]
        y_data = frame_data[:,-1].reshape(1, 1)
        return x_data, y_data

    def __getitem__(self, index):
        x_data, y_data = self.getframe(index)
        x_data_tensor = torch.tensor(x_data)
        y_data_tensor = torch.tensor(y_data)
        return x_data_tensor, y_data_tensor

    def __len__(self):
        return self.length

# file_path  = '/home/ubuntu/xlj/optimization/models/MNIST_01_image_lables.txt'
# mnist_dataset = MyDataset_mnist(file_path)
# img, lable = mnist_dataset[0] # 返回一个元组，返回值就是__getitem__的返回值


# # 获取整个训练集，就是对两个数据集进行了拼接
# train_dataset = mnist_dataset

# img1, label1 = train_dataset[123]  # 获取的是蚂蚁的最后一个
# img2, label2 = train_dataset[124]  # 获取的是蜜蜂第一个