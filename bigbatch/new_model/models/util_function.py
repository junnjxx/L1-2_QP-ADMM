import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# 对matrix的sample_order
#sample_order_df = pd.read_csv('models/mnist_01_bs_256_sample_order.csv')
# 对vector的sample_order
def covert_to_train_txt(ori_txt_dir, sample_order_dir, output_txt_dir):
    try:
        # 读取样本顺序
        sample_order_df = pd.read_csv(sample_order_dir, header=0)

        # 读取原始数据
        data_with_labels_df = pd.read_csv(ori_txt_dir, header=None)

        # 根据样本顺序对数据重新排序
        sorted_data_with_labels_df = data_with_labels_df.loc[sample_order_df['sample_order']]
        #sorted_data_with_labels_df = data_with_labels_df.iloc[sample_order_df.iloc[:, 0].values]
        # 将排序后的数据保存为新的文本文件
        sorted_data_with_labels_df.to_csv(output_txt_dir, sep=',', index=False)

        print("重新排序并保存完成！")
        return True
    except Exception as e:
        print("发生错误：", e)
        return False

class TwoLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # 使用xavier初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


import numpy as np
import random

def process_matrix(Y, bs, log_file):
    """
    Process a matrix Y, ensuring there are exactly `bs` elements equal to 1.

    Args:
        Y (numpy.ndarray): The input matrix.
        bs (int): The target number of elements equal to 1.
        log_file (str): Path to the log file for recording operations.

    Returns:
        numpy.ndarray: The processed indices of elements equal to 1.
    """
    # Convert to numpy array if necessary
    if isinstance(Y, torch.Tensor):
        Y = Y.cpu().numpy()

    # Find indices of elements equal to 1
    B = np.where(Y == 1)[0]

    # Open the log file in append mode
    with open(log_file, 'a') as log:
        # Log the original number of elements equal to 1
        log.write(f"Original count of elements equal to 1: {len(B)}, ")
        
        if len(B) < bs:
            # Randomly sample additional indices to supplement to `bs`
            additional_indices = random.choices(range(Y.shape[0]), k=bs - len(B))
            B = np.concatenate((B, additional_indices))
            log.write(f"Added {bs - len(B)} random indices to reach target size.\n")
        
        elif len(B) > bs:
            # Randomly select a subset of indices to reduce to `bs`
            B = np.random.choice(B, size=bs, replace=False)
            log.write(f"Reduced {len(B) - bs} indices to reach target size.\n")
        
        # Log the final number of elements equal to 1
        log.write(f"Final count of elements equal to 1: {len(B)}\n")
    
    return B

