from torch.utils.data import Dataset
import os
from PIL import Image
import csv
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import cv2


class ForgeryDataset(Dataset):
    def __init__(self, csv_path
                 , transforms=None):
        self.csv_path = csv_path
        self.transforms = transforms

        # 从CSV文件中读取数据
        self.image_paths = []
        self.labels = []
        self.masks = []
        with open(csv_path, mode='r') as file:
            lines = file.readlines()
            for line in lines[1:]:  # 跳过标题行
                parts = line.strip().split(',')
                self.image_paths.append(parts[0])
                self.labels.append(int(parts[1]))
                self.masks.append(parts[2])
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        label = self.labels[idx]
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0) / 255.0 

        else:
            image = transforms.ToTensor()(img)
            mask = transforms.ToTensor()(mask).unsqueeze(0) / 255.0  # 去掉通道维度
        # print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
        return image, mask, label

class ForgerySegDataset(Dataset):
    def __init__(self, csv_path
                 , transforms=None):
        self.csv_path = csv_path
        self.transforms = transforms

        # 从CSV文件中读取数据
        self.image_paths = []
        self.labels = []
        self.masks = []
        with open(csv_path, mode='r') as file:
            lines = file.readlines()
            for line in lines[1:]:  # 跳过标题行
                parts = line.strip().split(',')
                self.image_paths.append(parts[0])
                self.labels.append(int(parts[1]))
                self.masks.append(parts[2])
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        label = self.labels[idx]
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0) / 255.0 

        else:
            image = transforms.ToTensor()(img)
            mask = transforms.ToTensor()(mask).unsqueeze(0) / 255.0  # 去掉通道维度
        # print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
        return image, mask, label



class ForgerySegDataset(Dataset):
    def __init__(self, csv_path
                 , transforms=None):
        self.csv_path = csv_path
        self.transforms = transforms

        # 从CSV文件中读取数据
        self.image_paths = []
        self.masks = []
        with open(csv_path, mode='r') as file:
            lines = file.readlines()
            for line in lines[1:]:  # 跳过标题行
                parts = line.strip().split(',')
                self.image_paths.append(parts[0])
                self.masks.append(parts[1])
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0) / 255.0 

        else:
            image = transforms.ToTensor()(img)
            mask = transforms.ToTensor()(mask).unsqueeze(0) / 255.0  # 去掉通道维度
        # print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
        return image, mask



if __name__ == "__main__":

    IMG_SIZE = 512
    
    train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=90, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    csv_path = './ForgeryDataset.csv'
    dataset = ForgeryDataset(csv_path, transforms=train_transform)
    for img, mask, label in dataset:
        print(img.size, mask.size, label)
        break
    