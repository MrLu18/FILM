# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''
import os
import matplotlib.pyplot as plt
import sys
import time
import datetime
import torch
from utils.Logger import Logger1
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils.lossfun import Fusionloss, LpLssimLossweight
import numpy as np
from utils.H5_read import H5ImageTextDataset
import argparse
import warnings
from net.Film import Net
import logging
import shutil
import re
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.CRITICAL)

parser = argparse.ArgumentParser()


parser.add_argument('--i2t_dim', type=int, default=32, help='')
parser.add_argument('--hidden_dim', type=int, default=256, help='')
parser.add_argument('--numepochs', type=int, default=150, help='')
parser.add_argument('--lr', type=float, default=1e-5, help='') #学习率
parser.add_argument('--gamma', type=float, default=0.6, help='') #调整学习率的
parser.add_argument('--step_size', type=int, default=50, help='') #配合调整学习率的
parser.add_argument('--batch_size', type=int, default=2, help='')
parser.add_argument('--loss_grad_weight', type=int, default=20, help='')
parser.add_argument('--loss_ssim', type=int, default=0, help='')
parser.add_argument('--dataset_path', type=str, default="VLFDataset_h5\MSRS_train.h5", help='') #数据集路径
opt = parser.parse_args()

'''
------------------------------------------------------------------------------
Set the hyper-parameters for training
------------------------------------------------------------------------------
'''
pre_model = ""
num_epochs = opt.numepochs
lr = opt.lr
step_size = opt.step_size
gamma = opt.gamma
weight_decay = 0
batch_size = opt.batch_size
weight_ingrad = opt.loss_grad_weight
weight_ssim = opt.loss_ssim
hidden_dim = opt.hidden_dim
i2t_dim = opt.i2t_dim
dataset_path = opt.dataset_path
exp_name = '' #自己设置实验名称 可以没有

'''
------------------------------------------------------------------------------
model
------------------------------------------------------------------------------
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.nn.DataParallel(
    Net(hidden_dim=hidden_dim, image2text_dim=i2t_dim))
if pre_model != "": #有预训练模型就输出一下
    model.load_state_dict(torch.load(pre_model)['model'])
    print('load_pretrain_model')
model.to(device)
criterion = LpLssimLossweight().to(device) #这是损失函数（L1和SSIM组合）

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) #优化器设置
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) #学习率调整

trainloader = DataLoader(H5ImageTextDataset(dataset_path), batch_size=batch_size, #加载训练集 设置batch_size等
                         shuffle=True, num_workers=0, drop_last=True) #最后一个参数是指丢弃不完整批次
time_begin = time.strftime("%y_%m_%d_%H_%M", time.localtime()) #获取当前时间
save_path = "exp/" + str(time_begin) + '_epochs_%s' % (
    str(opt.numepochs)) + '_lr_%s' % (str(opt.lr)) + '_stepsize_%s' % (str(opt.step_size)) + '_bs_%s' % (
                           opt.batch_size) + '_gradweight_%s' % (str(opt.loss_grad_weight)) + '_gamma_%s' % (
                           str(opt.gamma)) + exp_name  #保存路径和名称 路径包括epochs 学习率 步长step_size batch_size 梯度权重 gamma参数
logger = Logger1(rootpath=save_path, timestamp=True)
params = {  #记录超参数 应该是用于保存的
    'epoch': num_epochs,
    'lr': lr,
    'batch_size': batch_size,
    'optim_step': step_size,
    'optim_gamma': gamma,
    'gradweight': weight_ingrad,
}
logger.save_param(params)
logger.new_subfolder('model') #创建子文件夹
writer = SummaryWriter(logger.logpath) #summaryWriter是pytorch tensorboard可视化工具 用于记录loss变化和accuracy的变化等
exp_folder = logger.get_timestamp_folder_name() #返回一个时间的文件夹名称 给exp_folder
destination_folder = os.path.join(save_path, exp_folder, 'code') #结合上面的save_folder


def save_code_files(source_file, destination_folder):
    global model_file_path
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    with open(source_file, 'r', encoding="utf-8") as file:
        content = file.read()
    match = re.search(r'from net\.(\w+) import Net', content) #这会从source_file中读取的内容寻找from net.xxx import net模式 查看是否匹配
    if match:
        model_name = match.group(1) #得到匹配模型的名称
        model_file_path = os.path.join('net', f'{model_name}.py')
    dest_train_file_path = os.path.join(destination_folder, os.path.basename(__file__))

    shutil.copyfile(source_file, dest_train_file_path)
    shutil.copyfile(model_file_path, os.path.join(destination_folder, f'{model_name}.py'))

save_code_files(os.path.basename(__file__), destination_folder)
'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''
step = 0
torch.backends.cudnn.benchmark = True #加速卷积操作
prev_time = time.time()
start_time = time.time()
loss = Fusionloss(coeff_grad=weight_ingrad, device=device)  #这个应该是自己定义的损失函数

for epoch in range(num_epochs):
    ''' train '''
    s_temp = time.time()
    model.train()
    for i, (data_IR, data_VIS, text, index) in enumerate(trainloader): #处理训练数据 这边是红外光和可见光 所以如果是MEF 应该是要修改一下内容
        data_VIS, data_IR, text = data_VIS.to(device), data_IR.to(device), text.to(device)
        text = text.squeeze(1).to(device)
        F = model(data_IR, data_VIS, text)
        batchsize, channels, rows, columns = data_IR.shape
        weighttemp = int(np.sqrt(rows * columns))
        lplssimA, lpA, lssimA = criterion(image_in=data_IR, image_out=F, weight=weighttemp) #计算IR的感知损失和结构性损失
        lplssimB, lpB, lssimB = criterion(image_in=data_VIS, image_out=F, weight=weighttemp) #计算VIS的
        loss_in_grad, _, _ = loss(data_IR, data_VIS, F) #计算融合的损失
        loss_ssim = lplssimA + lplssimB
        lossALL = loss_in_grad + weight_ssim * loss_ssim #权重损失是一开始有的
        optimizer.zero_grad() #清除之前的梯度信息
        lossALL.backward() #计算当前batch的梯度
        optimizer.step() #更新模型参数


        batches_done = epoch * len(trainloader) + i #计算剩余时间并打印参数
        batches_left = num_epochs * len(trainloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        logger.log_and_print(
            "[Epoch %d/%d] [Batch %d/%d] [loss: %f] [loss_in_grad: %f] [lplssimA: %f] [lplssimB: %f] ETA: %.10s"
            % (
                epoch + 1,
                num_epochs,
                i,
                len(trainloader),
                lossALL.item(),
                loss_in_grad.item(),
                lplssimA.item(),
                lplssimB.item(),
                time_left,
            )
        )

        #用tensorboard计算损失曲线
        writer.add_scalar('loss/01 Loss', lossALL.item(), step)
        writer.add_scalar('loss/01 loss_in_grad', loss_in_grad.item(), step)
        writer.add_scalar('loss/01 lplssimA', lplssimA.item(), step)
        writer.add_scalar('loss/01 lplssimB', lplssimB.item(), step)
        writer.add_scalar('loss/14 learning rate', optimizer.state_dict()['param_groups'][0]['lr'], step)
        step += 1


        if (epoch + 1) % 1 == 0:
            if i <= 1: #对于前两个epoch执行 减少计算开销
                for j in range(data_IR.shape[0]): #data_IR_shape[0]是batch_size 遍历全部

                    temp = np.zeros((rows, 3 * columns)) #生成一个numpy数组 应该是用于存储 三张图像的横向拼接

                    temp[:rows, 0:columns] = np.squeeze(data_IR[j].detach().cpu().numpy()) * 255 #去除第j个样本 转换为numpy 存入temp的第一列
                    temp[:rows, columns:columns * 2] = np.squeeze(data_VIS[j].detach().cpu().numpy()) * 255 # 同理 存入第二列
                    temp[:rows, columns * 2:columns * 3] = np.squeeze(F[j].detach().cpu().numpy()) * 255 #存入第三 就代表以此为 IR  VIS F

                    if not os.path.exists(os.path.join(logger.logpath, 'pic_fusion', "ckpt_" + str(epoch + 1))): #确保这个路径文件存在
                        os.makedirs(os.path.join(logger.logpath, 'pic_fusion', "ckpt_" + str(epoch + 1)))
                    plt.imsave(os.path.join(logger.logpath, 'pic_fusion', "ckpt_" + str(epoch + 1), #保存拼接之后的图像
                                            str(index[j]) + '.png'),
                               temp,
                               cmap="gray") #以灰度图形式存入此目录中

    scheduler.step() #调整学习率的 scheduler是学习率调度器
    # 保存模型
    if (epoch + 1) % 1 == 0: #每个epoch结束 保存一下
        checkpoint = {
            'model': model.state_dict(),
            'optimizer1': optimizer.state_dict(),
            'lr_schedule1': scheduler.state_dict(),
            "epoch": epoch,
            'step': step,
        }
        os.path.join(logger.logpath, 'model')
        torch.save(checkpoint, os.path.join(logger.logpath, 'model', 'ckpt_%s.pth' % (str(epoch + 1))))
    e_temp = time.time()
    print("This Epoch takes time: " + str(e_temp - s_temp))

end_time = time.time()
logger.log_and_print("total_time: " + str(end_time - start_time))
