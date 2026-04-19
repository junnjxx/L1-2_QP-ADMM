
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights, resnet18, resnet152

from util_function import TwoLayerMLP
from torch.utils.data import DataLoader
from my_datasets import MyDataset_mnist,MyDataset_cifar10,MyDataset_mnist_test,MyDataset_cifar10_test
from test_dl_models import run_test
from tensorboardX import SummaryWriter

import os
from datetime import datetime



def log_metrics_to_tensorboard(S, train_file_dir, test_file_dir, datasize, batch_size, device, model_type, dataset, optimizer_type,tb_writer,random_seed,num_repeats,num_epoch,lr):
    avg_train_losses, avg_test_accuracies , avg_test_loss = run_dl_models(S, train_file_dir, test_file_dir, datasize, batch_size, device, model_type, dataset,optimizer_type,random_seed,num_repeats,num_epoch,lr)
    
    for epoch, train_loss in enumerate(avg_train_losses):
        tb_writer.add_scalars('train_loss', {S: train_loss}, epoch + 1)

    for epoch, test_accuracy in enumerate(avg_test_accuracies):
        tb_writer.add_scalars('test_accuracy', {S: test_accuracy}, epoch + 1)
    
    for epoch, test_loss in enumerate(avg_test_loss):
        tb_writer.add_scalars('test_loss', {S: test_loss}, epoch + 1)

def run_dl_models(S,train_file,test_file,datasize,batch_size,device,model_type,dataset,optimizer_type,random_seed,num_repeats,num_epoch,init_lr):
    
    if S == 'Random':
        if dataset == 'mnist':
            train_data = MyDataset_mnist( train_file ,datasize,S,batch_size)
        elif dataset == 'cifar10':
            train_data = MyDataset_cifar10( train_file ,datasize,S,batch_size)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    if dataset == 'mnist':
        test_data = MyDataset_mnist_test(test_file, 10000)
    elif dataset == 'cifar10':
        test_data = MyDataset_cifar10_test(test_file, 10000)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False,pin_memory=True)

    criterion = nn.CrossEntropyLoss()

    # Training loop
    # 记录所有重复实验的训练损失和测试准确率
    all_train_losses = []
    all_test_accuracies = []
    all_test_loss = []
    # avg_train_losses = []
    # avg_test_accuracies = []
    # 进行多次重复训练
    for repeat in range(num_repeats):
        print(f'Repeat {repeat + 1}/{num_repeats}:')

        # 重置模型参数
        torch.manual_seed(random_seed)
        if model_type == '2mlp':
            model = TwoLayerMLP(input_size=1024,hidden_size=256,output_size=10)
        if model_type == 'resnet18':
                    # 加载预定义的 ResNet18 模型
            import torchvision.models as models
            resnet18 = models.resnet18()
            if dataset == 'mnist':
                i_c   =1
            elif dataset == 'cifar10':
                i_c  =3
            # 修改第一层卷积层
            resnet18.conv1 = nn.Conv2d(
                in_channels= i_c,       # 输入通道数 (CIFAR10 是 RGB 图像)
                out_channels=64,     # 输出通道数保持与原始模型一致
                kernel_size=3,       # 改为 3x3 卷积核
                stride=1,            # 步长设置为 1
                padding=1,           # 填充设置为 1，以保持输入尺寸
                bias=False           # 与原始 ResNet 一致，不使用偏置
            )

            # 如果需要使用更小的图片，可以去掉 maxpool
            resnet18.maxpool = nn.Identity()
             # 获取全连接层的输入特征数
            num_features = resnet18.fc.in_features

            # 添加 Dropout 层到全连接层之前
            resnet18.fc = nn.Sequential(
                nn.Dropout(p=0.5),       # Dropout 概率为 0.5
                nn.Linear(num_features, 10)  # CIFAR10 有 10 个类别
            )



            model = resnet18


        ###
        if model_type == 'resnet50':
            import torchvision.models as models
            resnet50 = models.resnet50()
            if dataset == 'mnist':
                i_c   =1
            elif dataset == 'cifar10':
                i_c  =3
            # 修改第一层卷积层
            resnet50.conv1 = nn.Conv2d(
                in_channels= i_c,       # 输入通道数 (CIFAR10 是 RGB 图像)
                out_channels=64,     # 输出通道数保持与原始模型一致
                kernel_size=3,       # 改为 3x3 卷积核
                stride=1,            # 步长设置为 1
                padding=1,           # 填充设置为 1，以保持输入尺寸
                bias=False           # 与原始 ResNet 一致，不使用偏置
            )

            # 如果需要使用更小的图片，可以去掉 maxpool
            resnet50.maxpool = nn.Identity()
             # 获取全连接层的输入特征数
            num_features = resnet50.fc.in_features

            # 添加 Dropout 层到全连接层之前
            resnet50.fc = nn.Sequential(
                nn.Dropout(p=0.5),       # Dropout 概率为 0.5
                nn.Linear(num_features, 10)  # CIFAR10 有 10 个类别
            )



            model = resnet50
        ####
        if model_type == 'resnet152':
            import torchvision.models as models
            resnet152 = models.resnet152()

            # 修改第一层卷积层
            resnet152.conv1 = nn.Conv2d(
                in_channels=1,       # 输入通道数 (CIFAR10 是 RGB 图像)
                out_channels=64,     # 输出通道数保持与原始模型一致
                kernel_size=3,       # 改为 3x3 卷积核
                stride=1,            # 步长设置为 1
                padding=1,           # 填充设置为 1，以保持输入尺寸
                bias=False           # 与原始 ResNet 一致，不使用偏置
            )

            # 如果需要使用更小的图片，可以去掉 maxpool
            resnet152.maxpool = nn.Identity()
            model = resnet152
        model = model.to(device)

        if optimizer_type == 'sgdw':
            optimizer = optim.SGD(model.parameters(), lr=init_lr, weight_decay=1e-4)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=init_lr)
        elif optimizer_type == 'sgdwm':
            optimizer = optim.SGD(model.parameters(),lr = init_lr,momentum=0.9, weight_decay=1e-4)
        elif optimizer_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=init_lr)

        # 定义学习率调度器，每 20 个 epoch 学习率衰减 0.1 倍
        scheduler = StepLR(optimizer, step_size = 30 , gamma=0.1) # adam
        scheduler = StepLR(optimizer, step_size = 20 , gamma=0.9)
        
        # 初始化当前重复实验的训练损失和测试准确率
        repeat_train_losses = []
        repeat_test_accuracies = []
        repeat_test_loss = []

        # 在每个 epoch 中训练模型
        for epoch in range(num_epoch):
            running_loss = 0.0
            if  S == 'Matrix':
                if dataset == "mnist":
                    train_data = MyDataset_mnist( train_file ,datasize,S,batch_size)
                    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
                elif dataset == 'cifar10':
                    train_data = MyDataset_cifar10( train_file ,datasize,S,batch_size)
                    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
            elif S == 'Vector':
                if dataset == 'mnist':
                    train_data = MyDataset_mnist( train_file ,datasize,S,batch_size)
                    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
                elif dataset == 'cifar10':
                    train_data = MyDataset_cifar10( train_file ,datasize,S,batch_size)
                    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)


            # 遍历训练集
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

                # 统计损失
                running_loss += loss.item()

            # 计算每个 epoch 的平均损失并将其添加到当前重复实验的列表中
            avg_epoch_loss = running_loss / len(train_loader)
            repeat_train_losses.append(avg_epoch_loss)


            # 更新学习率
            scheduler.step()

            # 打印学习率
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch + 1}/{num_epoch}, Learning Rate: {lr}")

            # 测试模型性能并计算准确率
            # 测试模型性能并计算准确率
            model.eval()  # 切换到评估模式
            test_accuracy , test_loss  = run_test(test_loader, model, epoch, criterion)
            model.train()  # 训练完后切换回训练模式

            # 将每个 epoch 的测试准确率添加到当前重复实验的列表中
            repeat_test_accuracies.append(test_accuracy)
            repeat_test_loss.append(test_loss)


        # 将当前重复实验的训练损失和测试准确率添加到所有实验的列表中
        all_train_losses.append(repeat_train_losses)
        all_test_accuracies.append(repeat_test_accuracies)
        all_test_loss.append(repeat_test_loss)

    # 计算所有实验的平均训练损失和平均测试准确率
    avg_train_losses = torch.tensor(all_train_losses).mean(dim=0)
    avg_test_accuracies = torch.tensor(all_test_accuracies).mean(dim=0)
    avg_test_loss = torch.tensor(all_test_loss).mean(dim=0)

    # # 将平均损失值和平均准确率写入 TensorBoard
    # for epoch, train_loss in enumerate(avg_train_losses):
    #     tb_writer.add_scalar(f'train_loss_{S}', train_loss, epoch+1)

    # for epoch, test_accuracy in enumerate(avg_test_accuracies):
    #     tb_writer.add_scalar(f'test_accuracy_{S}', test_accuracy, epoch+1)
    return avg_train_losses, avg_test_accuracies,avg_test_loss

def generate_log_dir(base_dir, model_type, dataset,datasize, lr, optimizer, decay, epochs, run_id,batch_size,random_seed):
    """
    Generate a dynamic log directory path based on model parameters.

    Args:
        base_dir (str): The base directory for logs.
        model_type (str): The type of model (e.g., '2mlp', 'resnet18').
        dataset (str): The dataset name (e.g., 'cifar10').
        lr (float): Learning rate.
        optimizer (str): Optimizer name (e.g., 'sgd', 'adam').
        decay (float): Weight decay value.
        epochs (int): Number of epochs.
        run_id (int): Run identifier to avoid overwriting.

    Returns:
        str: Generated log directory path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Add timestamp for uniqueness
    log_dir = os.path.join(
        base_dir,
        f"{dataset}_{str(datasize)}_{batch_size}_{model_type}_lr{lr}_decay{decay}_{optimizer}_epoch{epochs}_run{run_id}_randomseed_{random_seed}_{timestamp}"
    )
    return log_dir

# # Example usage
# base_dir = "/home/ubuntu/xlj/new_model/models/runs_mnist_59904_256"
# model_type = "resnet18"
# dataset = "mnist"
# datasize = 59904
# lr = 0.001
# optimizer_type = "sgdw"
# decay = 0.9
# num_epoch = 20
# num_repeats = 3
# batch_size = 256
# import random
# random_seed = random.randint(1, 1000)
# device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
# print(f"Using device: {device}")

# log_dir = generate_log_dir(base_dir, model_type, dataset,datasize, lr, optimizer_type, decay, num_epoch, num_repeats,random_seed)
# print(log_dir)

# tb_writer = SummaryWriter(log_dir )
# print(device)
# random_train_file_dir = "/home/ubuntu/xlj/new_model/Datasets/MNIST_jpg/train.txt"
# all_train_file_dir = "/home/ubuntu/xlj/new_model/datasets/train.txt"
# matrix_train_file_dir = "/home/ubuntu/xlj/new_model/Datasets/MNIST_jpg/60000_mnist_256/mnist_sorted_matrix_60000_256.txt"
# vector_train_file_dir = "/home/ubuntu/xlj/new_model/Datasets/MNIST_jpg/60000_mnist_256/mnist_sorted_vector_60000_256.txt"
# test_file_dir = '/home/ubuntu/xlj/new_model/Datasets/MNIST_jpg/testing.txt'


# # Logging for different setups

# log_metrics_to_tensorboard('Random', random_train_file_dir, test_file_dir, datasize, batch_size, device, model_type, dataset, optimizer_type,tb_writer,random_seed)
# log_metrics_to_tensorboard('Matrix', matrix_train_file_dir, test_file_dir, datasize, batch_size, device, model_type, dataset, optimizer_type,tb_writer,random_seed)
# log_metrics_to_tensorboard('Vector', vector_train_file_dir, test_file_dir, datasize, batch_size, device, model_type, dataset, optimizer_type,tb_writer,random_seed)

# 将平均损失值和平均准确率写入 TensorBoard
# 将平均损失值和平均准确率写入 TensorBoard
# 将训练损失放在同一张图上


# for epoch, (train_loss_random, train_loss_matrix, train_loss_vector) in enumerate(zip(avg_train_losses_random, avg_train_losses_matrix, avg_train_losses_vector)):
#     tb_writer.add_scalars('train_loss', {
#         'random': train_loss_random,
#         'matrix': train_loss_matrix,
#         'vector': train_loss_vector
#     }, epoch + 1)

# # 将测试准确率放在同一张图上
# for epoch, (test_accuracy_random, test_accuracy_matrix, test_accuracy_vector) in enumerate(zip(avg_test_accuracies_random, avg_test_accuracies_matrix, avg_test_accuracies_vector)):
#     tb_writer.add_scalars('test_accuracy', {
#         'random': test_accuracy_random,
#         'matrix': test_accuracy_matrix,
#         'vector': test_accuracy_vector
#     }, epoch + 1)


import random
import torch
from torch.utils.tensorboard import SummaryWriter

def setup_and_log_metrics(base_dir, model_type, dataset, datasize, lr, optimizer_type, decay, 
                          num_epoch, num_repeats, batch_size, random_seed=None):
    """
    Sets up logging and runs experiments for different training setups.

    Args:
        base_dir (str): Base directory for logs.
        model_type (str): Type of the model, e.g., 'resnet18'.
        dataset (str): Dataset name, e.g., 'mnist'.
        datasize (int): Size of the dataset.
        lr (float): Learning rate.
        optimizer_type (str): Optimizer type, e.g., 'sgdw'.
        decay (float): Weight decay or learning rate decay factor.
        num_epoch (int): Number of training epochs.
        num_repeats (int): Number of repetitions for experiments.
        batch_size (int): Batch size for training.
        random_seed (int, optional): Random seed for reproducibility. If not provided, one is generated.
    
    Returns:
        None
    """
    if random_seed is None:
        random_seed = random.randint(1, 1000)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    log_dir = generate_log_dir(base_dir, model_type, dataset, datasize, lr, optimizer_type, decay, 
                               num_epoch, num_repeats,batch_size, random_seed)
    print(f"Log directory: {log_dir}")

    tb_writer = SummaryWriter(log_dir)
    print(f"TensorBoard writer initialized: {device}")

    # for cifar-10
    random_train_file_dir = "/home/ubuntu/xlj/opt/bigbatch/cifar10_50000_250/train.txt"
    matrix_train_file_dir = "/home/ubuntu/xlj/opt/bigbatch/cifar10_50000_250_HOG/cifar10_sorted_matrix_50000_250_HOG.txt"
    vector_train_file_dir = "/home/ubuntu/xlj/opt/bigbatch/cifar10_50000_250_HOG/cifar10_sorted_vector_50000_250_HOG.txt"
    test_file_dir = "/home/ubuntu/xlj/opt/bigbatch/cifar10_50000_250/test.txt"
    #### for mnist
    # random_train_file_dir = "/home/ubuntu/xlj/opt/bigbatch/60000_mnist_256/train.txt"
    # matrix_train_file_dir = "/home/ubuntu/xlj/new_model/Datasets/MNIST_jpg/60000_mnist_256/mnist_sorted_matrix_60000_256.txt"
    # vector_train_file_dir = "/home/ubuntu/xlj/new_model/Datasets/MNIST_jpg/60000_mnist_256/mnist_sorted_vector_60000_256.txt"
    # test_file_dir = "/home/ubuntu/xlj/new_model/Datasets/MNIST_jpg/60000_mnist_256/mnist_test_corrected.txt"
    # Logging for different setups
    setups = {
        'Random': random_train_file_dir,
        'Matrix': matrix_train_file_dir,
        'Vector': vector_train_file_dir
    }

    for setup_name, train_file_dir in setups.items():
        log_metrics_to_tensorboard(setup_name, train_file_dir, test_file_dir, datasize, batch_size, 
                                   device, model_type, dataset, optimizer_type, tb_writer, random_seed,num_repeats,num_epoch,lr)



import argparse 
parser = argparse.ArgumentParser()
parser.add_argument("--optimizer_type", type=str, required=True,help="optimizer type, sgd 0.01 , sgdw w 1e-4, sgdwm m 0.9 , adam 0.0001 with decay 0.9 every 20 epochs")
parser.add_argument("--use_gpu", type=str,default="0")
parser.add_argument("--dataset_name", type=str,default="mnist")
parser.add_argument("--dataset_size", type=int,default="59904")
parser.add_argument("--bs", type=int,default="256")
parser.add_argument("--model_type", type=str,default="resnet18")
args = parser.parse_args()

### mnist 选的随机种子是81
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu  # 限制到特定 GPU
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
    for i in [8]:
        r_seed = 10 * i +1 
        setup_and_log_metrics(
            base_dir="/home/ubuntu/xlj/new_model/models/runs_cifar10_50000_250_HOG/epoch100",
            model_type = args.model_type,
            dataset= args.dataset_name,
            datasize=args.dataset_size,
            lr=0.1, # sgd都是0.1
            optimizer_type= args.optimizer_type,
            decay=0.1,  # sgd都是0.9，20 ， cifar10 adam 0.1 30
            num_epoch=100,
            num_repeats=3,
            batch_size=args.bs,
            random_seed= r_seed

        )
