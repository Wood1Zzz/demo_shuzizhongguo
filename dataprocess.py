import os
import csv
import re
import config

def create_csv(file_path):
    csv_filename = "ForgeryDataset.csv"
    csv_head = ["image_path", "label", "mask_path", "caption"]

    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_head)
    
    # 获取真实、伪造图像的路径
    fake = os.path.join(file_path, "Black", "Image")
    temp = os.listdir(fake)
    for i in temp:
        if i.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff')):
            mask_path = os.path.join(file_path, "Black", "Mask", i[:-4] + ".png")
            caption = os.path.join(file_path, "Black", "Caption", i[:-4] + ".md")
            with open(csv_filename, mode='a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([os.path.join(fake, i), 0, mask_path, caption])
    real = os.path.join(file_path, "White", "Image")
    temp = os.listdir(real)
    for i in temp:
        if i.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff')):
            mask_path = os.path.join(file_path, "White", "Mask", i[:-4] + ".png")
            caption = os.path.join(file_path, "White", "Caption", i[:-4] + ".md")
            with open(csv_filename, mode='a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([os.path.join(real, i), 1, mask_path, caption])

    print("CSV文件已创建并保存到当前目录:", csv_filename)


def create_seg_csv(file_path):
    csv_filename = "ForgerySegDataset.csv"
    csv_head = ["image_path", "mask_path"]

    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_head)
    
    # 获取真实、伪造图像的路径
    fake = os.path.join(file_path, "Black", "Image")
    temp = os.listdir(fake)
    for i in temp:
        if i.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff')):
            mask_path = os.path.join(file_path, "Black", "Mask", i[:-4] + ".png")
            with open(csv_filename, mode='a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([os.path.join(fake, i), mask_path])

    print("CSV文件已创建并保存到当前目录:", csv_filename)

if __name__ == "__main__":
    # dataset_path = config.DATASET_PATH
    # create_csv(dataset_path)
    dataset_path = config.DATASET_PATH
    create_seg_csv(dataset_path)
