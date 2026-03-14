import numpy as np
import os
from PIL import Image




# 获取png图像通道数
def get_png_channels(png_path):
    with Image.open(png_path) as img:
        return img.mode  # 返回图像模式，如 'RGB', 'RGBA', 'L' 等

# 转为RGB（丢弃透明通道，像素值与JPG一致）
def png_to_jpg_pixels(png_path):
    with Image.open(png_path) as img:
        rgb_img = img.convert('RGB')
        rgb_img.save(png_path, format='PNG')  # 覆盖保存为PNG格式

if __name__ == "__main__":

    data_path = '/root/autodl-tmp/demo_shuzizhongguo/ForgeryAnalysis_Stage_1_Train/Black/Image'
    print(f"Checking PNG channels in directory: {data_path}")

    print(f"Converting PNG to JPG-like pixels in directory: {data_path}")
    for file in os.listdir(data_path):
        if file.endswith('.png'):
            png_path = os.path.join(data_path, file)
            png_to_jpg_pixels(png_path)
            print(f"{file}: converted to RGB and saved as PNG")