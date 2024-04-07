import torch
import torchvision
import onnxruntime
from torchvision import transforms
import numpy as np
import config
# 加载 CIFAR-10 测试集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
testset = torchvision.datasets.CIFAR10(root='dataset', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=16)

# 加载 ONNX 模型
onnx_model_path = 'simclrv2_bm1684x.onnx'
ort_session = onnxruntime.InferenceSession(onnx_model_path)

# 获取输入和输出名称
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# 对测试集进行推理
correct = 0
total = 0

for data in testloader:
        images, labels = data
        images_np = images.numpy()
        outputs = ort_session.run([output_name], {input_name: images_np})
        _, predicted = torch.max(torch.from_numpy(outputs[0]), 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('准确率：%02.3f %%' % (100 * correct / total))