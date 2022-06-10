"""
本文件是想创建一个自动化脚本，对现有数据集进行分类
将图像中的男女打上标签，并将结果存储到 ./data/input/FacialBeautyPrediction/gender.csv
"""
import pandas as pd
from showm import show_img


if __name__ == "__main__":
    image_path = []
    gender = []
    data = pd.read_csv("./data/input/FacialBeautyPrediction/train.csv")
    for i in range(len(data)):
        w_path = data["image_path"][i]
        show_img("./data/input/FacialBeautyPrediction" + w_path[1:])
        image_path.append(w_path)
        gender.append(int(input(w_path + " 的性别: ")))

    # 将结果写入文件
    dataframe = pd.DataFrame({"image_path": image_path, "gender": gender})
    dataframe.to_csv("./data/input/FacialBeautyPrediction/gender.csv", index=False, sep=',')
