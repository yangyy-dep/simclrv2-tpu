# SimCLRv2

## 目录

* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型与数据](#3-准备模型与数据)
* [4. 模型编译](#4-模型编译)
* [5. 例程测试](#5-例程测试)
* [6. 精度测试](#6-精度测试)
  * [6.1 测试方法](#61-测试方法)
  * [6.2 测试结果](#62-测试结果)
* [7. 性能测试](#7-性能测试)
  * [7.1 bmrt_test](#71-bmrt_test)
  * [7.2 程序运行性能](#72-程序运行性能)
    
## 1. 简介
本例程对[torchvision SimCLRv2](https://github.com/google-research/simclr)的模型和算法进行移植，使之能在SOPHON BM1684X 上进行推理测试。

**论文:** [SimCLRv2论文](https://arxiv.org/abs/2006.10029)

SimCLRv2由图灵奖获得者Geoffrey Hinton博士指导的Google大脑研究人员团队推出，这种方法，在充分利用大量未标记数据的同时，从少数标记示例中学习的一种范式是无监督预训练，然后是监督微调，该方法对 ImageNet 的半监督学习非常有效。

在此非常感谢Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, Geoffrey Hinton等人的贡献。

## 2. 特性
* 支持BM1684X SoC
* 支持FP32、FP16(BM1684X)、INT8模型编译和推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持图片测试

## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel，Pytorch模型在编译前要导出成onnx模型。

同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

模型包括：
```
.
├── BM1684X
│   ├── simclrv2_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── simclrv2_fp32_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=4
│   ├── simclrv2_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
    ├── simclrv2_fp16_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=4
│   ├── simclrv2_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── simclrv2_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
├── torch
│   └── simclrv2.pth             # 原始模型
│   
└── onnx
    └── simclrv2_bm1684x.onnx                          # 导出的动态onnx模型
```

数据包括：
```
./datasets
└── test                   # 测试图片, 共10000张   
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台，如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x 
```

​执行上述命令会在`models/BM1684X`等文件夹下生成`simclrv2_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台，如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x 
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成`simclrv2_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台，如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684x 
```

​上述脚本会在`models/BM1684`等文件夹下生成`simclrv2_int8_1b.bmodel`等文件，即转换好的INT8 BModel。

## 5. 例程测试
* [Python例程](python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改相关参数。  
然后，使用`eval_test.py`脚本，将预测结果文件与测试集标签文件进行对比，计算出分类准确率。具体的测试命令如下：
```bash
# 请根据实际情况修改文件路径
python3 eval_test.py 
```
### 6.2 测试结果
在imagenet_val_1k数据集上，精度测试结果如下：
|   测试平台    |      测试程序       |        测试模型        | ACC(%) |
| ------------ | ----------------   | ---------------------- | ------ |
| BM1684X SoC | simclrv2.py  | simclrv2_fp16_1b.bmodel  | 89.25  |
| BM1684X SoC | simclrv2.py  | simclrv2_fp32_1b.bmodel  | 89.26  |
| BM1684X SoC | simclrv2.py  | simclrv2_int8_1b.bmodel  | 89.23  |
| BM1684X SoC | simclrv2.py  | simclrv2_int8_4b.bmodel  | 89.23  |


> **测试说明**：  
> 1. batch_size=4和batch_size=1的模型精度一致；
> 2. 由于sdk版本之间可能存在差异，实际运行结果与本表有<1%的精度误差是正常的；
## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684x/simclrv2_fp16_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型      | calculate time(ms) |
| ----------------------------- | ----------------- |
| BM1684X/simclrv2_fp16_1b.bmodel | 1.79              |
| BM1684X/simclrv2_fp32_1b.bmodel | 3.69              |
| BM1684X/simclrv2_int8_1b.bmodel | 1.48              |
| BM1684X/simclrv2_int8_4b.bmodel | 0.50              |


### 7.2程序运行性能
查看统计的解码时间、预处理时间、推理时间、后处理时间。python例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/test`，性能测试结果如下：
|    测试平台  |     测试程序        |        测试模型       |preprocess_time|inference_time|postprocess_time| 
| ----------- | ------------------ | --------------------- | --------- | --------- | --------- |
| BM1684X SoC  | simclrv2.py   | simclrv2_fp16_1b.bmodel | 0.14      | 2.07      | 0.17      |
| BM1684X SoC  | simclrv2.py   | simclrv2_fp32_1b.bmodel | 0.15      | 3.98      | 0.17      |
| BM1684X SoC  | simclrv2.py   | simclrv2_int8_1b.bmodel | 0.15      | 1.85      | 0.18      |
| BM1684X SoC  | simclrv2.py   | simclrv2_int8_4b.bmodel | 0.10      | 0.81      | 0.11      |


> **测试说明**：  
1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
2. 性能测试结果具有一定的波动性，建议多次测试取平均值；


