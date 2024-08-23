

# Jittor 开放域少样本视觉分类赛题  lora微调结合多种adapter的半监督少样本分类方法

## 简介
| 简单介绍项目背景、项目特点

本项目包含了第四届计图挑战赛计图 - 开放域少样本视觉分类赛题的代码实现。
1.采用了 clip-lora方法对clip模型微调处理。
2.结合tip-adapter和AMU-Tuning训练分类头。
3.对生成的伪标签数据（每个类别16张）再加入训练集，重复训练分类头。
4.最终在A榜测试集取得了最高72.37%的效果。

## 安装 
| 介绍基本的硬件需求、运行环境、依赖安装方法

本项目可在24G显卡条件上运行，训练时间约为30分钟。

#### 运行环境
- ubuntu 20.04
- python == 3.8
- jittor == 1.3.8

#### 安装依赖
执行以下命令安装 python 依赖
```
pip install -r requirements.txt
```

#### 预训练模型下载并转换为pkl

本项目需要两个预训练模型：
1.clip的ViT-B-32.pkl版本，需要放到代码根目录下。
2.amu的预训练模型，预训练模型模型下载地址为 https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar，
下载到根目录，需要通过转换代码zh.py，将r-50-1000ep.pth.tar转换为jittor版本，r-50-1000ep.pkl。

## 数据集下载解压


将数据下载并解压到 `<root>/caches` 下，执行以下命令对数据预处理：

1.训练集caches/TrainSet
2.测试集caches/TestSetA


## 训练

单卡训练可运行以下命令：
```
bash train.sh
```
不支持多卡训练


## 推理

测试集结果在训练完成自动生成，存放在result/result.txt。
