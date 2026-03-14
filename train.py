import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import csv
from dataset import ForgeryDataset, ForgerySegDataset
from model import deeplabv3_resnet50
import config
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from lossFunction import dice_loss, combined_loss, calculate_iou
from tqdm import tqdm
from sklearn.model_selection import KFold
import pandas as pd
import glob
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR



def main():
    df = pd.read_csv('./ForgerySegDataset.csv')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    train_transform = A.Compose([
    A.Resize(config.IMG_SIZE, config.IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),  # 随机旋转
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=90, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    ToTensorV2()
    ], is_check_shapes=False)

    val_transform = A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], is_check_shapes=False)

    # 加载模型、优化器和学习率调度器
    print("Loading DeepLabV3 with ImageNet Weights...")
    model = deeplabv3_resnet50(weights='DEFAULT')
    print("✅ Loaded ImageNet Weights!")
    print("✅ Using device:", config.DEVICE)
    
    model.classifier[4] = nn.Sequential(
        nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
        nn.ReLU(),
        nn.Dropout(p=0.5),  # 加入 Dropout
        nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    )

    model.aux_classifier[4] = nn.Sequential(
        nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
        nn.ReLU(),
        nn.Dropout(p=0.5),  # 加入 Dropout
        nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    )

    model = model.to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    best_iou = 0.0
    best_fold = 0
    for fold, (train_index, val_index) in enumerate(kf.split(df)):
        print(f"--- Fold {fold+1} ---")
        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]

        # 保存临时csv文件
        train_df.to_csv(f'train_fold.csv', index=False)
        val_df.to_csv(f'val_fold.csv', index=False)

        train_dataset = ForgerySegDataset('train_fold.csv', transforms=train_transform)
        val_dataset = ForgerySegDataset('val_fold.csv', transforms=val_transform)

        train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        patience = 5  # 容忍5个epoch
        counter = 0

        # TRAINING LOOP (With IoU Display) ---
        print(f"--- STARTING TRAINING (DeepLabV3 + IoU Tracking) ---")
        best_epoch = 1
        for epoch in range(config.EPOCHS):
            model.train()
            epoch_loss = 0.0
            epoch_iou = 0.0

            loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")

            for images, masks in loop:
                images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
                optimizer.zero_grad()
                outputs = model(images)['out']

                loss = combined_loss(outputs, masks.float())
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # Calculate IoU for this batch
                batch_iou = calculate_iou(outputs, masks.float())
                epoch_iou += batch_iou

                # Update Progress Bar with current IoU
                loop.set_postfix(loss=loss.item(), iou=batch_iou)

            avg_train_loss = epoch_loss / len(train_dataloader)
            avg_train_iou = epoch_iou / len(train_dataloader)
                
            # validate 
            model.eval()
            val_loss = 0.0
            val_iou = 0.0
            with torch.no_grad():
                for val_images, val_masks in val_dataloader:  # Using train_dataloader for validation as well
                    val_images, val_masks = val_images.to(config.DEVICE), val_masks.to(config.DEVICE)
                    val_outputs = model(val_images)['out']
                    
                    val_loss += combined_loss(val_outputs, val_masks.float()).item()
                    val_iou += calculate_iou(val_outputs, val_masks.float())

                    avg_val_loss = val_loss / len(val_dataloader)
                    avg_val_iou = val_iou / len(val_dataloader)

            print(f"Epoch [{epoch+1}/{config.EPOCHS}] - Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}, Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}")

            # Update learning rate
            scheduler.step()

            if avg_val_iou > best_iou:
                best_iou = avg_val_iou
                best_epoch = epoch + 1
                best_fold = fold + 1
                counter = 0
                torch.save(model.state_dict(), f'deeplab_seg_best_model.pth')
                print(f"✅ New Best Model Saved with Val IoU: {best_iou:.4f} at Fold {best_fold}, Epoch {best_epoch}")
            else:
                counter += 1
                print(f"EarlyStopping counter: {counter}/{patience}")
                if counter >= patience:
                    print("⏹️ Early stopping triggered!")
                    break

    print("Training Completed!")
    print(f"Best Validation IoU: {best_iou:.4f}")
    print(f"✅ New Best Model Saved with Val IoU: {best_iou:.4f}")
    print(f"📌 Best Model is from Fold {best_fold}, Epoch {best_epoch}")

    save_count = 0
    with torch.no_grad():
        for val_images, val_masks in val_dataloader:
            val_images, val_masks = val_images.to(config.DEVICE), val_masks.to(config.DEVICE)
            val_outputs = model(val_images)['out']
            
            val_loss += combined_loss(val_outputs, val_masks.float()).item()
            val_iou += calculate_iou(val_outputs, val_masks.float())

            # 保存部分验证结果
            if save_count < 5:  # 仅保存前5个样本
                img = val_images[0].cpu()
                true_mask = val_masks[0].cpu().squeeze(0)
                pred_logits = val_outputs[0].cpu().squeeze(0)
                pred_mask = (torch.sigmoid(pred_logits) > 0.5).float()
                
                # 转换为PIL图像
                to_pil = transforms.ToPILImage()
                img_pil = to_pil(img)
                true_mask_pil = to_pil(true_mask.unsqueeze(0))
                pred_mask_pil = to_pil(pred_mask.unsqueeze(0))
                
                if os.path.exists('temp') == False:
                    os.makedirs('temp')

                # 保存到temp文件夹
                img_pil.save(f'temp/val_image_{save_count}.png')
                true_mask_pil.save(f'temp/true_mask_{save_count}.png')
                pred_mask_pil.save(f'temp/pred_mask_{save_count}.png')
                
                save_count += 1

            avg_val_loss = val_loss / len(val_dataloader)
            avg_val_iou = val_iou / len(val_dataloader)
    

    
if __name__ == "__main__":
    main()
