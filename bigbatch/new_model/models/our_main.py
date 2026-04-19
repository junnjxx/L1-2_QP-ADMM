import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from my_datasets import  MyDataset_cifar10_test
from our_models import Matirx_method, Vector_method
import numpy as np
import os 
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import gc
import datetime
import time
import yaml
import gc , shutil
from util_function import covert_to_train_txt,process_matrix
import argparse
import pandas as pd
##### save all configs in a txt #############
save_config_dir = 'saved_config_week_5_13_vector.txt' 
parser = argparse.ArgumentParser()
debug = 0
if debug:
    custom_save_path = "vector_58/"
    save_config_help = 'debug'
else:
    parser.add_argument("--custom_save_path", type=str, required=True,help="save_path_for_matrix, like matrix_4_30_6/")
    parser.add_argument('--save_config_help',type = str, required=True,help='each exper config will be saved in a unify txt, please write down the reason about why last exper stopped and new exper starts')
    parser.add_argument("--config_dir",default= '/home/ubuntu/xlj/new_model/models/configs/mnist_vector.yaml', help= 'config dir ' ) 
    parser.add_argument('--test_each_iter_time', default= 1 )
    args = parser.parse_args()
    custom_save_path = args.custom_save_path
    save_config_help = args.save_config_help

config_dir = args.config_dir
test_each_iter_time = args.test_each_iter_time
# config_dir  =  '/home/ubuntu/xlj/optimization/models/configs/mnist_vector.yaml'
# custom_save_path = 'demo1/'
# test_each_iter_time =1
with open(config_dir, "r") as file:
    config = yaml.safe_load(file)
t1 = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = config['experiment_info']["cuda_visible_devices"]
device = torch.device(config['experiment_info']["device_type"] if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
bandwidth_type = config['experiment_info']['bandwidth_type']
eta = config['model_parameters']['eta']
beta = config['model_parameters']['beta']
bs = config['model_parameters']['batch_size']
eta_inc = config['model_parameters']['eta_incr']
epsi = float(config["model_parameters"]["epsi"])
n_iter = config["model_parameters"]["n_iter"]
iter = int(float(config["model_parameters"]["iter_per_loop"]))
save_dir = os.path.join(config['output_files']['save_directory'], custom_save_path)
output_path = os.path.join(save_dir, config['output_files']['middle_info_save_dir'])
save_X_Y_name = config['output_files']['save_X_Y']
final_Y_save_dir = os.path.join(save_dir, config['output_files']['final_Y_save_dir'])
sample_order_save_dir = os.path.join(save_dir,config['output_files']['sample_order_save_dir'] )
data_size = config['data_info']['data_size']

if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)
## M and V dataloader
experiment_type = config['experiment_info']['type']
if experiment_type == 'Matrix' or 'Vector':
    dataset = MyDataset_cifar10_test(config['data_info']['dataset_path'],data_size)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
else:
    dataset = MyDataset_cifar10_test(config['data_info']['dataset_path'],data_size)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
imageMatrix = []
for images, _ in data_loader:
    imageVector = images.view(1, -1)
    imageMatrix.append(imageVector)

imageMatrix = torch.cat(imageMatrix, dim=0)

#### special for cifar10_HOG 
# 读取 CSV 文件
csv_file_path = '/home/ubuntu/xlj/opt/bigbatch/cifar10_HOG_feature.csv'  # 替换为你的 CSV 文件路径
data = pd.read_csv(csv_file_path, header=None)  # 假设没有列名
# 将 pandas DataFrame 转为 PyTorch 张量
imageMatrix = torch.tensor(data.values, dtype=torch.float32)
# 打印结果，查看维度
print("imageMatrix shape:", imageMatrix.shape)

####
pairwiseNorms = torch.cdist(imageMatrix, imageMatrix, p=2)
if bandwidth_type == 'mean':
    bandwidth = torch.sum(pairwiseNorms) / (imageMatrix.size(0) ** 2 - imageMatrix.size(0))
del imageMatrix

phi0 = torch.exp(-pairwiseNorms / bandwidth)
eta = torch.tensor(eta)
beta = torch.tensor(beta)
n, k = phi0.size()
with open(config_dir, "r") as f:
    yaml_content = f.read()
with open(output_path, "w") as f:
    f.write(yaml_content)
with open(save_config_dir, "a+") as f:
    f.write(save_config_help)
    f.write('\n')
    f.write(yaml_content)
    


if  experiment_type  == 'Matrix':
    nb = n // bs
    k = n
    n = nb
    e = torch.ones(k,n)
    G = phi0 @ e

    # 只取可整除的
    phi0 = phi0[:n*bs, :n*bs] * 2 / bs / bs
    G = G[:n*bs, :] * -4 / bs / k
    eta = eta.to(device)
    beta = beta.to(device)
    A = phi0.to(device)
    G = G.to(device)
    n_,k_ =G.size()
    X = torch.rand(n_, k_).to(device)
    Y  = torch.rand(n_, k_).to(device)
    Lambda = torch.zeros(n_, k_).float().to(device)
    # X = pd.read_csv("/home/ubuntu/xlj/optimization/models/resume/Matrix_0_X_less20_9624.csv",header = None)
    # Y = pd.read_csv("/home/ubuntu/xlj/optimization/models/resume/Matrix_0_Y_less20_9624.csv",header= None)
    # Lambda = pd.read_csv("/home/ubuntu/xlj/optimization/models/resume/Matrix_0_LAMBDA_less20_9624.csv",header=None)
    # X = X.to_numpy()
    # X = torch.tensor(X).to(device)
    # Y = Y.to_numpy()
    # Y = torch.tensor(Y).to(device)
    # Lambda = Lambda.to_numpy()
    # Lambda = torch.tensor(Lambda).to(device)
    e = e.to(device)

    gc.collect()
    torch.cuda.empty_cache()
    for i in range(n_iter):
        save_X_Y_dir = f"{save_X_Y_name}_{i}"
        save_X_Y_dir = os.path.join(save_dir, save_X_Y_dir)
        X, Y, error_list = Matirx_method(A, G, eta, beta, iter, X, Y, Lambda, epsi ,device, i ,output_path,test_each_iter_time,save_X_Y_dir )
        torch.save(X, f"{save_X_Y_dir}_X.pt")
        torch.save(X, f"{save_X_Y_dir}_Y.pt")
       


    Y =Y.cpu().numpy()
    # 使用 torch.savetxt 函数将张量保存为 CSV 文件
    np.savetxt(final_Y_save_dir, Y, delimiter=",")

    M = []
    for i in range(k_):
        m_temp = np.where(Y[:, i])[0]
        M.append(m_temp)

    M_ = np.array(M)
    M_transpose = M_.T
    Sample_order = M_transpose.flatten()
    Sample_order +=1
    np.savetxt(sample_order_save_dir, Sample_order, fmt='%d', delimiter=',')
    t2 = time.time()
    t1_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1))
    t2_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t2))
    # 打印转换后的时间字符串
    print("Start time:", t1_time)
    print('end time:',t2_time)
    print("total time: {:.2f}second".format(t2 - t1))

    with open(output_path, "a") as f:
        f.write(f"Start time: {t1_time}\n")
        f.write(f"End time: {t2_time}\n")
        f.write(f"Total time: {t2 - t1:.2f} seconds\n")

    print(f"sample order 已保存到文件: {sample_order_save_dir}")

elif experiment_type == 'Vector':

    t1=time.time()
    nb = n // bs
    # 创建索引向量D
    D = np.arange(0, nb*bs )
    n = nb*bs
    save_B = []
    Q = D.copy()
    B = []
    P = []
    s_B =[]
    saved_Y =[]
    ### 读取 save_B 的，并调整 Q P
    #save_B_dir = "/home/ubuntu/user/xlj/new_model/models/save_1023_vector/vector1022/save_B.csv"
    save_B_dir = 0
    if save_B_dir:
    
        # 从 CSV 文件中读取 resume_B
        resume_B = np.loadtxt(save_B_dir, delimiter=',')
        
        # 确保 Q 和 resume_B 是 numpy 数组
        Q = np.array(Q)  # 假设 Q 已经初始化为 numpy 数组
        resume_B = np.array(resume_B)
        
        # 使用 np.setdiff1d 从 Q 中去除 resume_B，并保持 Q 排序
        Q = np.sort(np.setdiff1d(Q, resume_B))

        # 可选：根据需要调整 P，或做进一步处理
        temp_k = len(resume_B) // bs
        resume_B_matrix = resume_B.reshape(temp_k, bs)

        # 转换为列表，每行代表一个长度为 bs 的子列表
        resume_B_list = [resume_B_matrix[i, :] for i in range(temp_k)]
        P = resume_B_list
        P = np.array(P)


       

        
    for i in range(nb):
        if save_B_dir:
            if i > 1:
                P = np.vstack((P, s_B))
        else:
            if i  <=1:
                P = s_B
            else:
                P = np.vstack((P, s_B))
        
        Q = np.sort(np.setdiff1d(Q, s_B))
        n_p = len(P)  * bs  #n_p = P.shape[1]
        print(n_p)
        nq = Q.shape[0]
        phi1 = phi0[Q[:, None], Q].numpy()
        phi2 = np.sum(phi1, axis=0) * (n_p + bs) / n
        if n_p > 0:  
            P_flat = P.ravel()
            phi_qp = phi0[Q[:, None], P_flat]
            phi3 = np.sum(phi_qp.numpy(), axis=1) * (nq - bs) / n
        Z = phi1.copy()
        for j in range(nq):
            if n_p > 0:
                Z[j, j] -= phi2[j] - phi3[j]
            else:
                Z[j, j] -= phi2[j]
        X = torch.rand(nq,1).to(device)
        Y = torch.rand(nq,1).to(device)
        for j in range(n_iter ):
            save_X_Y_dir = f"{save_X_Y_name}_{i}_{j}"
            save_X_Y_dir = os.path.join(save_dir, save_X_Y_dir)
            Z =torch.tensor(Z).to(device)
            X , Y, error_list = Vector_method(Z, n_p,nq,bs,eta, beta, iter, X,Y,epsi,device, i,j ,output_path, test_each_iter_time,save_X_Y_dir)
            # torch.save(X, f"{save_X_Y_dir}_X.pt")
            # torch.save(X, f"{save_X_Y_dir}_Y.pt")
          
        # Y = Y.cpu().numpy()
        # B = np.where(Y == 1)[0] 
        colum_dis_dir = 'vector_log_1125.txt'
        B = process_matrix(Y,bs,colum_dis_dir)
        s_B = Q[B] 
        save_B = np.concatenate((save_B, s_B)) 
        np.savetxt(os.path.join(save_dir,'save_B.csv'), save_B, delimiter=',')

    left_Q = np.setdiff1d(Q, s_B)
    save_B = np.concatenate((save_B, left_Q))

    M_ = np.array(save_B)
    M_transpose = M_.T
    Sample_order = M_transpose.flatten()
    # Sample_order +=1
    np.savetxt(sample_order_save_dir, Sample_order, fmt='%d', delimiter=',')
    t2 = time.time()
    t1_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1))
    t2_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t2))
    print("Start time:", t1_time)
    print('end time:',t2_time)
    print("total time: {:.2f}second".format(t2 - t1))

    with open(output_path, "a") as f:
        f.write(f"Start time: {t1_time}\n")
        f.write(f"End time: {t2_time}\n")
        f.write(f"Total time: {t2 - t1:.2f} seconds\n")

    print(f"sample order saved!: {sample_order_save_dir}")
    with open(colum_dis_dir, 'a', ) as file:
        file.write('\ntraining end or stopped\n')

# 打开原始文件并读取内容
with open(sample_order_save_dir, "r") as file:
    content = file.readlines()

# 在内容的开头插入新行
new_line = "sample_order\n"
content.insert(0, new_line)

# 将修改后的内容写回文件
with open(sample_order_save_dir, "w") as file:
    file.writelines(content)

### 根据sample_order 生成 训练的txt文件
ge_train_txt = config['experiment_info']['train_txt']
print(ge_train_txt)
if ge_train_txt:
    train_txt_dir = config['output_files']['final_train_txt_dir']
    train_txt_dir = os.path.join(save_dir,train_txt_dir )
    success = covert_to_train_txt(config['data_info']['dataset_path'], sample_order_save_dir, train_txt_dir  )

    