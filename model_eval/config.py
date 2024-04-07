# config.py
import os
from torchvision import transforms

use_gpu=True
gpu_name=0

#pre_model=os.path.join('pth','model.pth')
pre_model='pth1/model_stage1_epoch1000.pth'
pe_model='pth1/model_stage2_epoch200.pth'
p_model='pth1/model_stage2_epoch165.pth'
save_path="pth1"
#数据预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),#随机裁剪图像到32*32
    transforms.RandomHorizontalFlip(p=0.5),#以50%的概率水平翻转图像
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),#以80%的概率对图像应用颜色抖动操作，包括亮度、对比度、饱和度和色相的随机变化。
    transforms.RandomGrayscale(p=0.2),#以20%的概率将图像转换为灰度图像
    transforms.ToTensor(),#将图像转换为张量（Tensor）格式，以便能够输入到神经网络中
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])#先将像素值转换为范围在 0 到 1 之间的浮点数，然后再进行减去均值除以标准差
#所以/255?
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
#对于每个通道，将图像的像素值减去对应通道的均值（[0.4914, 0.4822, 0.4465]）。
#然后，将结果除以对应通道的标准差（[0.2023, 0.1994, 0.2010]）
