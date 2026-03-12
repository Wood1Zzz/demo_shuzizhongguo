import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import timm 
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from sklearn.model_selection import KFold
import os
import numpy as np
import tqdm



EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
IMG_SIZE = 224
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ForgeryClassDataset(Dataset):
    def __init__(self, csv_path, transforms=None):
        self.csv_path = csv_path
        self.transforms = transforms

        # 从CSV文件中读取数据
        self.image_paths = []
        self.labels = []
        with open(csv_path, mode='r') as file:
            lines = file.readlines()
            for line in lines[1:]:  # 跳过标题行
                parts = line.strip().split(',')
                self.image_paths.append(parts[0])
                self.labels.append(int(parts[1]))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transforms:
            image = self.transforms(img)
        else:
            image = transforms.ToTensor()(img)
        return image, label


# loss function
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class ForgeryClassifier(nn.Module):
    def __init__(self):
        super(ForgeryClassifier, self).__init__()
        self.model = timm.create_model('resnet50', pretrained=True)
        # 替换最后一层为输出1个logit
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)

def main():
    df = pd.read_csv('./ForgeryDataset.csv')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = ForgeryClassifier().to(DEVICE)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    for fold, (train_index, val_index) in enumerate(kf.split(df)):
        print(f"--- Fold {fold+1} ---")
        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]

        # 保存临时csv文件
        train_df.to_csv(f'train_cls_fold.csv', index=False)
        val_df.to_csv(f'val_cls_fold.csv', index=False)

        train_dataset = ForgeryClassDataset(f'train_cls_fold.csv', transforms=train_transform)
        val_dataset = ForgeryClassDataset(f'val_cls_fold.csv', transforms=val_transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        patience = 5  # 容忍5个epoch
        counter = 0

        # TRAINING LOOP (With IoU Display) ---
        print(f"--- STARTING TRAINING (ResNet50 Tracking) ---")
        best_epoch = 1
        min_loss = float('inf')
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            loop = tqdm.tqdm(train_loader, desc=f"Fold {fold+1} Training")
            
            for images, labels in loop:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

            # 验证阶段
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    loss = criterion(outputs.squeeze(), labels.float())
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f'Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}')
            scheduler.step()
            # Early Stopping Logic
            if avg_val_loss < min_loss:
                min_loss = avg_val_loss
                best_epoch = epoch + 1
                counter = 0  # 重置计数器
                torch.save(model.state_dict(), 'class_best_model.pth')
                print(f"New best model saved at epoch {best_epoch} with val loss {min_loss:.4f}")
            else:
                counter += 1
                print(f"No improvement. Counter: {counter}/{patience}")
                if counter >= patience:
                    print("Early stopping triggered.")
                    break


if __name__ == "__main__":
    main()