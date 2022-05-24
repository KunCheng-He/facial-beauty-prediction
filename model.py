import torch
import torchvision
from torch import nn
from torch.nn import functional as F


# 残差块的定义
class Residual(nn.Module):
    """
    input_channels 是我们的输入通道数，num_channels 是我们的输出通道数，strides 默认为 1
    当我们不改变步长与通道数时，可以不使用旁路的卷积层
    如果主线的输出形状改变了，那就需要使用旁路的卷积层来改变输入的输出，让最后可以按位置相加
    """

    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, 
                               padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
            Y += X  # 位置一一对应相加
        return F.relu(Y)


# 构建残差块
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    # first_block 为真，代表是第一个残差块，只有第一个残差块不使用旁路 1*1卷积，其余都用，这里只改变通道，不改变图像大小
    # first_block 为假，后面的通用残差块，第一节使用旁路，输出大小为 （输入x + 1)/2 ，第二节，不改变图像大小
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

# 初始化模型参数
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

# 返回 Resnet 模型
def myResnet():
    # 第一部分
    b1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 通道：3 -> 64  形状：(300-7+3*2+2)/2=150
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 通道不变  形状：(150-3+1*2+2)/2=75
    )
    # 第二部分
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))  # 通道不变  形状不变
    # 第三部分
    b3 = nn.Sequential(*resnet_block(64, 128, 2))  # 通道: 64 -> 128  形状：(75+1)/2=38
    # 第四部分
    b4 = nn.Sequential(*resnet_block(128, 256, 2))  # 通道：128 -> 256  形状：(38+1)/2=19
    # 第五部分
    b5 = nn.Sequential(*resnet_block(256, 512, 2))  # 通道：256 -> 512  形状：(19+1)/2=10
    # 最后的网络（最后一层我做了改变，10*10 求平均我觉得太大了，再加一层卷积，每个通道改到3*3，后面全连接输出维度下降太快了，使用了一个简单的 MPL）
    net = nn.Sequential(
        b1, b2, b3, b4, b5,
        nn.Conv2d(512, 512, kernel_size=3, stride=3),  # 通道不变  形状：(10-3+3)/3=3
        nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),  # 每个通道上 3*3 求平均
        nn.Flatten(),  # 512*1=512
        nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32),  # 加的一个简单 MLP
        nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1)
    )
    net.apply(init_weights)  # 初始化模型参数
    return net

# 通过微调再试试 resnet 模型
def fnResnet():
    # 加载网络上训练好的模型
    pretrained_net = torchvision.models.resnet18(pretrained=True)
    # 修改最后一个线性层，我们输出维度为 1
    pretrained_net.fc = nn.Linear(pretrained_net.fc.in_features, 1)
    nn.init.xavier_uniform_(pretrained_net.fc.weight)
    return pretrained_net


if __name__ == "__main__":
    # 测试一下残差块的输出
    X = torch.rand(1, 3, 224, 224)  # 先初始化一个输入
    net = myResnet()
    # print("X shape: ", X.shape)
    # print("X out shape: ", net(X).shape)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, "输出形状: ", X.shape)
