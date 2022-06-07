# -*- coding: utf-8 -*
from flyai_sdk import FlyAI
from PIL import Image
import torchvision
from model import myResnet, myResnet2, myResnet3, myResnet4, myResnet34
import torch


class Prediction(FlyAI):
    def __init__(self) -> None:
        self.net = self.load_model()

    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        # net 为我们本次训练好的模型，要和载入的参数相对应
        # net = myResnet()
        # net = myResnet2()
        # net = myResnet3()
        net = myResnet4()
        # net = myResnet34()
        net.load_state_dict(torch.load("myResnet4.params"))  # 载入对应训练好的模型
        return net

    def predict(self, data):
        '''
        模型预测返回结果
        :param input: 评估传入样例 data为要预测的图片路径. 示例 "./data/input/image/0.jpg"
        :return: 模型预测成功中, 直接返回预测的结果 ，返回示例 浮点类型 如3.11
        '''
        img = Image.open(data)
        augs = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(  # 统一裁剪为 224*224，区域覆盖原来的 90% 以上
                (224, 224), scale=(0.9, 1), ratio=(1, 1)  # 高宽比不变（脸型还是很重要的，不要变形）
            ),
            torchvision.transforms.ToTensor()
        ])
        img = augs(img)
        img = img.reshape(1, 3, 224, 224)
        return self.net(img).item()


# 以下是我写的本地测试部分，提交到平台上将以下注释掉即可
# if __name__ == "__main__":
#     pre = Prediction()
#     while True:
#         id = input("编号: ")
#         if id == "0":
#             break
#         print(pre.predict("./data/input/FacialBeautyPrediction/image/" + id + ".jpg"))
