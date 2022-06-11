# -*- coding: utf-8 -*-
import argparse
import math
import os
import sys


from flyai_sdk import FlyAI, DataHelper, MODEL_PATH, DATA_PATH

import json
import pandas as pd

# 我自己导入的包
from torch.utils.data import Dataset, sampler, DataLoader
from PIL import Image
import torch
from torch import nn
import numpy as np
import torchvision
from showm import apply as ap  # 自己显示图像的 API ，看一下增强后的图像
from showm import show_img
import numpy as np
from model import myResnet, myResnet2, myResnet3, myResnet4, Resnet34



'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 项目的超参，不使用可以删除
# parser = argparse.ArgumentParser()
# parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
# parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
# args = parser.parse_args()


# 尝试用GPU
def try_gpu(i=0):
    """如果存在 GPU(i) ，就返回，否则返回 cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


# 加载数据集的类
class MyDataset(Dataset):
    def __init__(self, data, transform = None, target_transform = None) -> None:
        self.imgs = []
        for i in range(len(data)):
            score = data["label"][i]
            self.imgs.append((
                DATA_PATH + "/FacialBeautyPrediction" + data["image_path"][i][1:],
                score.astype(np.float32)
            ))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''

    def download_data(self):
        # 根据数据ID下载训练数据
        data_helper = DataHelper()
        data_helper.download_from_ids("FacialBeautyPrediction")

    def deal_with_data(self, batch_size):
        data = pd.read_csv("./data/input/FacialBeautyPrediction/train.csv")  # 读入文件
        augs = torchvision.transforms.Compose([  # 做一下图像增强
            torchvision.transforms.RandomHorizontalFlip(p=0.5),  # 随机左右翻转
            torchvision.transforms.RandomResizedCrop(  # 统一裁剪为 224*224，区域覆盖原来的 90% 以上
                (224, 224), scale=(0.9, 1), ratio=(1, 1)  # 高宽比不变（脸型还是很重要的，不要变形）
            ),
            torchvision.transforms.ToTensor()
        ])
        train_data = MyDataset(data, transform=augs)
        valid_rate = 0.2  # 验证集的比例
        # 下面这一段是对数据进行打乱
        data_size = len(train_data)
        indices = list(range(data_size))
        split = int(data_size * valid_rate)
        np.random.shuffle(indices)
        train_indices, valid_indices = indices[split:], indices[:split]
        train_sampler = sampler.SubsetRandomSampler(train_indices)
        valid_sampler = sampler.SubsetRandomSampler(valid_indices)
        # 以上随机序列已经生成好，下面载入数据的时候按随机序列载入就可以了
        train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
        valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)
        return train_loader, valid_loader

    # 计算总损失
    def sum_loss(self, net, data_iter, loss, device):
        l , n = 0.0, 0.0
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            y = y.reshape(y_hat.shape)
            l += loss(y_hat, y).item()
            n += len(y)
        return l, n

    def train(self, net, train_iter, valid_iter, epochs, lr, device):
        # 打印训练的设备
        print("train on : ", device)

        # 优化器使用 Adam，损失函数用均方误差
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        loss = nn.MSELoss()

        # 把模型迁移到指定设备上
        net.to(device)

        # 开始训练
        for epoch in range(epochs):
            for X, y in train_iter:
                # 迁移到设备
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                y = y.reshape(y_hat.shape)
                l = loss(y_hat, y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
            
            # 一个 epoch 已经训练完了，我们看看在训练集上的损失，和验证集上的损失
            trainLoss, trainNum = self.sum_loss(net, train_iter, loss, device)
            validLoss, validNum = self.sum_loss(net, valid_iter, loss, device)
            print("epoch {} -- train {} loss: {:.6f} -- valid {} loss: {:.6f}".format(epoch+1, trainNum, trainLoss, validNum, validLoss))

        # 返回我们训练完成的模型
        return net
    

if __name__ == '__main__':
    # 模型的一些参数
    batch_size = 8
    epochs = 30
    # 各个模型我所使用的 学习率
    # lr = 1e-4  # myResnet
    # lr = 5e-5  # myResnet2
    # lr = 8e-5  # myResnet3
    # lr = 5e-5  # myResnet4
    lr = 1e-4  # Resnet34
    device = try_gpu()

    # 主程序
    main = Main()
    main.download_data()  # 本地第一次运行需要，下载后本地可将本行代码注释掉，提交到平台上这行代码不要注释
    train_loader, valid_loader = main.deal_with_data(batch_size)
    
    # 开始训练
    net = main.train(Resnet34(), train_loader, valid_loader, epochs, lr, device)

    # 将我们训练的模型保存下来
    model_name = 'Resnet34.params'
    torch.save(net.state_dict(), MODEL_PATH + '/' + model_name)
    print("保存模型到" + MODEL_PATH + '/' + model_name)

    # 单步程序，训练完成后看看具体某张图片的打分（可以在Prediction.py文件中测试）
    # while True:
    #     id = input("请输入图像编号: ")
    #     path = DATA_PATH + "/FacialBeautyPrediction/image/" + id + ".jpg"
    #     img = Image.open(path)
    #     augs = torchvision.transforms.Compose([
    #         torchvision.transforms.RandomResizedCrop(  # 统一裁剪为 300*300，区域覆盖原来的 90% 以上
    #             (300, 300), scale=(0.9, 1), ratio=(1, 1)  # 高宽比不变（脸型还是很重要的，不要变形）
    #         ),
    #         torchvision.transforms.ToTensor()
    #     ])
    #     img = augs(img)
    #     img = img.reshape(1, 3, 300, 300)
    #     img = img.to(device)
    #     print("该图像的预测得分: ", net(img).item())
        