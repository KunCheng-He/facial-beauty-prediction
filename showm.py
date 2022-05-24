import matplotlib.pyplot as plt
from PIL import Image

def show_img(img, show_axis="off"):
    """
    显示单张图像

    可以传入图像的路径，也可以是 PIL 图像
    """
    if isinstance(img, str):
        img = Image.open(img)
    if show_axis == "off":
            plt.axis('off')
    plt.imshow(img)
    plt.show()

def show_imgs(imgs, num_rows, num_cols, show_axis="off"):
    """
    显示多张图像

    imgs - list - 包含我们要显示的图像
    num_rows - int - 行数
    num_rows - int - 列数
    show_axis - 默认不显示刻度线
    """
    idx = 1
    for img in imgs:
        if isinstance(img, str):
            img = Image.open(img)
        plt.subplot(num_rows, num_cols, idx)
        if show_axis == "off":
            plt.axis('off')
        plt.imshow(img)
        idx += 1
    plt.show()

def apply(img, aug, num_rows, num_cols):
    """对图像增强，并显示出来"""
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_imgs(Y, num_rows, num_cols)
