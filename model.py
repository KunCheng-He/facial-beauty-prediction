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
            blk.append(Residual(input_channels, num_channels,
                       use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


# 初始化模型参数
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


# 原版 resnet18
def Resnet18():
    # 第一部分
    b1 = nn.Sequential(
        # 通道：3 -> 64  形状：(224-7+3*2+2)/2=112
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), nn.ReLU(),
        # 通道不变  形状：(112-3+1*2+2)/2=56
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    # 第二部分
    b2 = nn.Sequential(
        *resnet_block(64, 64, 2, first_block=True))  # 通道不变  形状不变
    # 第三部分
    # 通道: 64 -> 128  形状：(56+1)/2=28
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    # 第四部分
    # 通道：128 -> 256  形状：(28+1)/2=14
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    # 第五部分
    # 通道：256 -> 512  形状：(14+1)/2=7
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    # 最后的网络
    net = nn.Sequential(
        b1, b2, b3, b4, b5,
        nn.AdaptiveAvgPool2d((1, 1)),  # 每个通道上 7*7 求平均
        nn.Flatten(), nn.Linear(512, 1)  # 我改的一个 MLP
    )
    net.apply(init_weights)  # 初始化模型参数
    return net


# 返回 Resnet 模型
def myResnet():
    # 第一部分
    b1 = nn.Sequential(
        # 通道：3 -> 64  形状：(224-7+3*2+2)/2=112
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), nn.ReLU(),
        # 通道不变  形状：(112-3+1*2+2)/2=56
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    # 第二部分
    b2 = nn.Sequential(
        *resnet_block(64, 64, 2, first_block=True))  # 通道不变  形状不变
    # 第三部分
    # 通道: 64 -> 128  形状：(56+1)/2=28
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    # 第四部分
    # 通道：128 -> 256  形状：(28+1)/2=14
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    # 第五部分
    # 通道：256 -> 512  形状：(14+1)/2=7
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    # 最后的网络（最后一层我做了改变，10*10 求平均我觉得太大了，再加一层卷积，每个通道改到3*3，后面全连接输出维度下降太快了，使用了一个简单的 MPL）
    net = nn.Sequential(
        b1, b2, b3, b4, b5,
        nn.AdaptiveAvgPool2d((1, 1)),  # 每个通道上 7*7 求平均
        nn.Flatten(),  # 512*1=512
        nn.Linear(512, 128), nn.ReLU(), nn.Linear(
            128, 8), nn.ReLU(), nn.Linear(8, 1)  # 我改的一个 MLP
    )
    net.apply(init_weights)  # 初始化模型参数
    return net


# 疯狂改进型2
def myResnet2():
    # 第一部分
    b1 = nn.Sequential(
        # 通道：3 -> 64  形状：(224-7+3*2+2)/2=112
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), nn.ReLU(),
        # 通道不变  形状：(112-3+1*2+2)/2=56
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    # 第二部分
    b2 = nn.Sequential(
        *resnet_block(64, 64, 2, first_block=True))  # 通道不变  形状不变
    # 第三部分
    # 通道: 64 -> 128  形状：(56+1)/2=28
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    # 第四部分
    # 通道：128 -> 256  形状：(28+1)/2=14
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    # 第五部分
    # 通道：256 -> 512  形状：(14+1)/2=7
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    # 最后的网络（最后一层我做了改变，7*7 求平均我觉得太大了，再加一层卷积，每个通道改到3*3，后面全连接输出维度下降太快了，使用了一个简单的 MPL）
    net = nn.Sequential(
        b1, b2, b3, b4, b5,
        nn.Conv2d(512, 512, kernel_size=3, stride=3),  # 通道不变  形状：(7-3+3)/3=2
        nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),  # 每个通道上 2*2 求平均
        nn.Flatten(),  # 512*1=512
        # 加的一个每次折半的 MLP
        nn.Linear(512, 256), nn.ReLU(), nn.Linear(
            256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(), nn.Linear(
            32, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1)
    )
    net.apply(init_weights)  # 初始化模型参数
    return net


# 暴改第 3 版
def myResnet3():
    # 第一部分
    b1 = nn.Sequential(
        # 通道：3 -> 64  形状：(224-7+3*2+2)/2=112
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), nn.ReLU(),
        # 通道不变  形状：(112-3+1*2+2)/2=56
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    # 第二部分
    b2 = nn.Sequential(
        *resnet_block(64, 64, 2, first_block=True))  # 通道不变  形状不变
    # 第三部分
    # 通道: 64 -> 128  形状：(56+1)/2=28
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    # 第四部分
    # 通道：128 -> 256  形状：(28+1)/2=14
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    # 第五部分
    # 通道：256 -> 512  形状：(14+1)/2=7
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    # 最后的网络（最后一层我做了改变，7*7 求平均我觉得太大了，再加一层卷积，每个通道改到3*3，后面全连接输出维度下降太快了，使用了一个简单的 MPL）
    net = nn.Sequential(
        b1, b2, b3, b4, b5,
        nn.Conv2d(512, 512, kernel_size=3, stride=3),  # 通道不变  形状：(7-3+3)/3=2
        nn.ReLU(), nn.Conv2d(512, 512, kernel_size=2),  # 通道不变 形状：2-2+1=1
        nn.Flatten(),  # 512*1=512
        # 加的一个每次折半的 MLP
        nn.Linear(512, 256), nn.ReLU(), nn.Linear(
            256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(), nn.Linear(
            32, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1)
    )
    net.apply(init_weights)  # 初始化模型参数
    return net


# 暴改第 4 版
def myResnet4():
    # 第一部分
    b1 = nn.Sequential(
        # 通道：3 -> 64  形状：(224-7+3*2+2)/2=112
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), nn.ReLU(),
        # 通道不变  形状：(112-3+1*2+2)/2=56
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    # 第二部分
    b2 = nn.Sequential(
        *resnet_block(64, 64, 2, first_block=True))  # 通道不变  形状不变
    # 第三部分
    # 通道: 64 -> 128  形状：(56+1)/2=28
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    # 第四部分
    # 通道：128 -> 256  形状：(28+1)/2=14
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    # 第五部分
    # 通道：256 -> 512  形状：(14+1)/2=7
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    # 最后的网络（最后一层我做了改变，7*7 求平均我觉得太大了，再加一层卷积，每个通道改到3*3，后面全连接输出维度下降太快了，使用了一个简单的 MPL）
    net = nn.Sequential(
        b1, b2, b3, b4, b5,
        nn.Conv2d(512, 512, kernel_size=3, stride=3),  # 通道不变  形状：(7-3+3)/3=2
        nn.ReLU(), nn.Conv2d(512, 512, kernel_size=2),  # 通道不变 形状：2-2+1=1
        nn.Flatten(),  # 512*1=512
        # 加的一个 MLP
        nn.Linear(512, 128), nn.ReLU(), nn.Linear(
            128, 8), nn.ReLU(), nn.Linear(8, 1)
    )
    net.apply(init_weights)  # 初始化模型参数
    return net


# 更深的 Resnet 模型 Resnet34
def Resnet34():
    # 第一部分
    b1 = nn.Sequential(
        # 通道：3 -> 64  形状：(224-7+2*3+2)/2=112
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), nn.ReLU(),
        # 通道：64  形状：(112-3+2*1+2)/2=56
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    # 第二部分
    b2 = nn.Sequential(
        *resnet_block(64, 64, 3, first_block=True))  # 通道：64  形状：56
    # 第三部分
    # 通道：64 -> 128 形状：(56+1)/2=28
    b3 = nn.Sequential(*resnet_block(64, 128, 4))
    # 第四部分
    # 通道：128 -> 256 形状：(28+1)/2=14
    b4 = nn.Sequential(*resnet_block(128, 256, 6))
    # 第五部分
    # 通道：256 -> 512 形状：(14+1)/2=7
    b5 = nn.Sequential(*resnet_block(256, 512, 3))
    # 最后的网络
    net = nn.Sequential(
        b1, b2, b3, b4, b5,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(), nn.Linear(512, 1)
    )
    net.apply(init_weights)  # 初始化模型参数
    return net


# 默认的 Resnet34 评分一直为 0 分，去掉全局平均池化层试试
def myResnet34():
    # 第一部分
    b1 = nn.Sequential(
        # 通道：3 -> 64  形状：(224-7+2*3+2)/2=112
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), nn.ReLU(),
        # 通道：64  形状：(112-3+2*1+2)/2=56
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    # 第二部分
    b2 = nn.Sequential(
        *resnet_block(64, 64, 3, first_block=True))  # 通道：64  形状：56
    # 第三部分
    # 通道：64 -> 128 形状：(56+1)/2=28
    b3 = nn.Sequential(*resnet_block(64, 128, 4))
    # 第四部分
    # 通道：128 -> 256 形状：(28+1)/2=14
    b4 = nn.Sequential(*resnet_block(128, 256, 6))
    # 第五部分
    # 通道：256 -> 512 形状：(14+1)/2=7
    b5 = nn.Sequential(*resnet_block(256, 512, 3))
    # 最后的网络
    net = nn.Sequential(
        b1, b2, b3, b4, b5,
        nn.Conv2d(512, 512, kernel_size=3, stride=3),  # 通道：512  形状：(7-3+3)/3=2
        nn.ReLU(), nn.Conv2d(512, 512, kernel_size=2),  # 通道：512  形状：2-2+1=1
        nn.Flatten(), nn.Linear(512, 1)
    )
    net.apply(init_weights)
    return net

"""
-----------------------------------------------------
以上两个 Resnet34 的模型，在训练时，损失下降是正常的，但是，
放到预测集上进行预测的时候，分数偏低，集中在 0~1 这个范围。
导致 MSE 很高，从而导致模型评分为 0，一下换稠密连接网络试试
-----------------------------------------------------
"""

# 稠密块中所需要的卷积块
def conv_block(input_channels, num_channels):
    # 通道：改为输出通道  大小：x-3+2*1+1=x 尺寸不会改变
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
    )

# 稠密块
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels) -> None:
        super().__init__()
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add_module(str(_), conv_block(input_channels, num_channels))
            # 下一个块的通道数要加上上一个块的输出通道数
            input_channels += num_channels
    
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 通道维度做叠加
            X = torch.cat((X, Y), dim=1)
        return X

# 稠密连接网络中所用的 过渡层（稠密网络会使通道数不断的增加，过渡层来控制通道数）
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),  # 通过一个 1*1 的卷积核来改变通道数
        nn.AvgPool2d(kernel_size=2, stride=2)  # (x-2+2)/2=x/2  形状大小减半
    )

# 构建一个稠密连接网络
def Densenet():
    net = nn.Sequential()
    # 第一部分，和 Resnet 一样
    net.add_module("start_block", nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 通道：3 -> 64  形状：(224-7+2*3+2)/2=112
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 通道：64  形状：(112-3+2*1+2)/2=56
    ))
    # 中间的稠密块部分，我们也分为了 4 块，每个稠密块里有 4 个卷积块，其中前三个稠密块之后加上过渡层来控制通道数
    input_channels, growth_rate = 64, 32  # 一个当前通道数，一个是每个卷积后新加的通道数
    for i, num_convs in enumerate([4, 4, 4, 4]):
        net.add_module("DenseBlock{}".format(i), DenseBlock(num_convs, input_channels, growth_rate))
        # 上一个稠密块输出的通道数
        input_channels += num_convs * growth_rate
        # 在稠密块之间加入过渡层控制通道数
        if i != 3:
            net.add_module("transition_block{}".format(i), transition_block(input_channels, int(input_channels/2)))
            input_channels = int(input_channels/2)
    # 最后的结尾部分  上一部分最后的输出：通道：248  形状：7*7
    net.add_module("end_block", nn.Sequential(
        nn.BatchNorm2d(248), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(), nn.Linear(248, 1)
    ))
    net.apply(init_weights)
    return net


if __name__ == "__main__":
    # 测试一下网络的输出
    X = torch.rand(1, 3, 224, 224)  # 先初始化一个输入
    # 想测试哪个模型就让 net 为指定模型就行
    net = Densenet()
    # print(net(X).shape)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, "输出形状: ", X.shape)
