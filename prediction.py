# -*- coding: utf-8 -*
from flyai_sdk import MODEL_PATH, FlyAI, DATA_PATH
from PIL import Image
import torchvision
from model import Resnet18, myResnet, myResnet2, myResnet3, myResnet4, Resnet34, myResnet34, Densenet, myDensenet
import torch


class Prediction(FlyAI):
    def __init__(self) -> None:
        self.net = self.load_model()

    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        model_name = "myDensenet.params"
        # net 为我们本次训练好的模型，要和载入的参数相对应
        # net = myResnet()
        # net = myResnet2()
        # net = myResnet3()
        # net = myResnet4()
        # net = myResnet34()
        # net = Densenet()
        net = myDensenet()
        # 打印导入模型的路径
        print("导入训练完成的模型：" + MODEL_PATH + '/' + model_name)
        net.load_state_dict(torch.load(
            MODEL_PATH + '/' + model_name))  # 载入对应训练好的模型
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
        score = self.net(img).item()
        print(data, " ---> ", score)
        # 手动对越界分数做限定
        if score < 0:
            return 0.0
        elif score > 5:
            return 5.0
        else:
            return score


# 以下是我写的本地测试部分，提交到平台上将以下注释掉即可
# if __name__ == "__main__":
#     pre = Prediction()
#     while True:
#         id = input("编号: ")
#         if id == "-1":
#             break
#         print(pre.predict(DATA_PATH + "/FacialBeautyPrediction/image/" + id + ".jpg"))
