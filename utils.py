'''
这个脚本的功能是从指定的输入图像路径中读取所有的图像文件
（支持.jpg和.png格式），然后为每个图像创建一个全黑的图像，
并将其保存到指定的输出路径中。输出图像的格式为.png，
文件名与输入图像相同但扩展名不同。
'''
from PIL import Image
import numpy as np
import os
import config

def create_black_image(input_image_path, output_image_path):
    for img_name in os.listdir(input_image_path):
        if img_name.endswith('.jpg') or img_name.endswith('.png'):
            input_image = Image.open(os.path.join(input_image_path, img_name))
            width, height = input_image.size
            # mode = input_image.mode  # 获取颜色模式，如 'RGB', 'L' (灰度) 等

            # 创建全黑图像（所有像素值为0）,'L'表示灰度图像
            black_image = Image.new('L', (width, height), color=0)

            # 保存黑色图像
            black_image.save(os.path.join(output_image_path, img_name.replace('.jpg', '.png')))
    print("黑色图像已创建并保存到:", output_image_path)

if __name__ == "__main__":
    color_image_path = config.DATASET_PATH + "/White/Image"
    black_image_path = config.DATASET_PATH + "/White/Mask"
    if not os.path.exists(black_image_path):
        os.makedirs(black_image_path)
    
    create_black_image(color_image_path, black_image_path)