import os
import torch
import torch.nn.functional as F
import torch.nn as nn
#gpu_id = int(os.environ['LOCAL_RANK'])
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device(0 )
print(f"Device: {device}")
#model.load_state_dict(torch.load(m_path + m_name, map_location={config.TRAIN.DEVICE: config.TEST.DEVICE}))
#dataloader_test = DataLoader(test_my(input_dir , gt_dir ),batch_size = 1, shuffle = False, pin_memory= False)

import torch.nn.functional as F

def run_test(dataloader, model, epoch, criterion):
    """
    运行测试集，计算准确率和测试损失。

    参数:
    - dataloader: 测试集数据加载器
    - model: 测试的模型
    - epoch: 当前的 epoch 数
    - criterion: 损失函数

    返回:
    - acc_value: 测试集的准确率
    - test_loss: 测试集的平均损失
    """
    model.eval()
    model = model.to(device)
    
    acc_number = 0
    total_loss = 0
    total_samples = 0

    for iter, data in enumerate(dataloader):  
        img = data[0].to(device)
        label = data[1].to(device)
        total_samples += label.size(0)

        with torch.no_grad():
            try:
                output = model(img)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise exception

        # 计算损失
        loss = criterion(output, label)
        total_loss += loss.item() * label.size(0)  # 按批次累积损失

        # 计算准确率
        probs = F.softmax(output, dim=1)
        _, predicted = torch.max(probs, 1)
        acc_number += (predicted == label).sum().item()

    acc_value = acc_number / total_samples
    test_loss = total_loss / total_samples

    print(f'Epoch={epoch}, acc_value={acc_value:.5f}, test_loss={test_loss:.5f}')
    return acc_value, test_loss
    
def run_test_sy_linear(dataloader,model,epoch):
    model.eval()
    model = model.to(device)
    mses = 0
    criterion = nn.MSELoss(reduction='mean')
    for iter, data  in enumerate(dataloader):  
        img = data[0].cuda()
        lable = data[1].cuda()
        img=img.to(torch.float32)
        with torch.no_grad():
            try:
                output = model(img)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception
        mses += criterion(output.float(), lable.float())
            
            
    avg_mses = mses/(len(dataloader))
    print('Epoch=%d,avg_mses=%.5f\n'%(epoch,avg_mses))
    return avg_mses