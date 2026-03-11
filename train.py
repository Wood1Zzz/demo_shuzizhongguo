import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import csv
from dataset import ForgeryDataset
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



def main():
    df = pd.read_csv(config.CSV_PATH)
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

    best_iou = 0.0
    for fold, (train_index, val_index) in enumerate(kf.split(df)):
        print(f"--- Fold {fold+1} ---")
        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]

        # 保存临时csv文件
        train_df.to_csv(f'train_fold.csv', index=False)
        val_df.to_csv(f'val_fold.csv', index=False)

        train_dataset = ForgeryDataset('train_fold.csv', transforms=train_transform)
        val_dataset = ForgeryDataset('val_fold.csv', transforms=val_transform)

        train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)



        train_dataset = ForgeryDataset(config.CSV_PATH, transforms=train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
        # plt.subplot(1, 2, 1)
        # img, mask, label = next(iter(dataloader))
        # plt.imshow(transforms.ToPILImage()(img[0]))
        # plt.subplot(1, 2, 2)
        # plt.imshow(transforms.ToPILImage()(mask[0].squeeze(0)))
        # plt.show()
        
        if fold == 0:  # 仅在第一个fold加载预训练权重，后续fold继续训练
            try:
                print("Loading DeepLabV3 with ImageNet Weights...")
                model = deeplabv3_resnet50(weights='DEFAULT')
                print("✅ Loaded ImageNet Weights!")
            except:
                model = deeplabv3_resnet50(weights=None)
        # else:
            # model = deeplabv3_resnet50(weights='./deeplab_best_model.pth')
        
        print("✅ Using device:", config.DEVICE)
        model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

        model = model.to(config.DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

        patience = 5  # 容忍5个epoch
        counter = 0

        # TRAINING LOOP (With IoU Display) ---
        print(f"--- STARTING TRAINING (DeepLabV3 + IoU Tracking) ---")
        for epoch in range(config.EPOCHS):
            model.train()
            epoch_loss = 0.0
            epoch_iou = 0.0

            loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")

            for images, masks, label in loop:
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
                for val_images, val_masks, val_labels in val_dataloader:  # Using train_dataloader for validation as well
                    val_images, val_masks = val_images.to(config.DEVICE), val_masks.to(config.DEVICE)
                    val_outputs = model(val_images)['out']
                    
                    val_loss += combined_loss(val_outputs, val_masks.float()).item()
                    val_iou += calculate_iou(val_outputs, val_masks.float())

                    avg_val_loss = val_loss / len(val_dataloader)
                    avg_val_iou = val_iou / len(val_dataloader)

            print(f"Epoch [{epoch+1}/{config.EPOCHS}] - Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}, Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}")

            if avg_val_iou > best_iou:
                best_iou = avg_val_iou
                counter = 0
                torch.save(model.state_dict(), f'deeplab_best_model.pth')
                print(f"✅ New Best Model Saved with Val IoU: {best_iou:.4f}")
            else:
                counter += 1
                print(f"EarlyStopping counter: {counter}/{patience}")
                if counter >= patience:
                    print("⏹️ Early stopping triggered!")
                    break

    print("Training Completed!")

    save_count = 0
    with torch.no_grad():
        for val_images, val_masks, val_labels in val_dataloader:
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
    
    test_dir = "/home/danzer/Documents/code/demo_shuzizhongguo/ForgeryAnalysis_Stage_1_Test/Image"
    output_dir = "/home/danzer/Documents/code/demo_shuzizhongguo/inference_results"
    infer_and_save(model, test_dir, output_dir, 'deeplab_best_model.pth', val_transform)

def infer_and_save(model, test_dir, output_dir, pth, transform):
    """
    对测试目录中的图片进行推理，并保存预测结果。

    Args:
        model: 训练好的模型。
        test_dir: 测试图片所在目录。
        output_dir: 推理结果保存目录。
        transform: 测试图片的预处理变换。
    """
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
    
    state_dict = torch.load(pth)
    # 将参数载入模型
    model.load_state_dict(state_dict)
    model = model.to(config.DEVICE)  # 确保模型在正确的设备上
    model.eval()  # 设置模型为评估模式
    to_pil = transforms.ToPILImage()

    # 获取所有图片路径
    image_paths = glob.glob(os.path.join(test_dir, "*.png")) + glob.glob(os.path.join(test_dir, "*.jpg"))

    with torch.no_grad():
        for idx, image_path in enumerate(image_paths):
            # 加载图片
            image = Image.open(image_path).convert("RGB")
            h, w = image.size
            transformed = transform(image=np.array(image))
            image_tensor = transformed["image"].unsqueeze(0).to(config.DEVICE)

            # 推理尺寸与输入一致
            # output = model(image_tensor)['out']
            # pred_mask = (torch.sigmoid(output[0]) > 0.5).float().cpu()
            # pred_mask = F.interpolate(pred_mask.unsqueeze(0), size=(w, h), mode='nearest').squeeze(0)
            # 推理尺寸与输入不一致
            output = model(image_tensor)['out']
            pred_mask = (torch.sigmoid(output[0])).float().cpu()
            pred_mask = (pred_mask > 0.5).float()  # 二值化掩码
            
            

            # 转换为PIL图像
            pred_mask_pil = to_pil(pred_mask.squeeze(0))  # 修复：移除多余的维度
            image_name = os.path.basename(image_path)

            # 保存原始图片和预测掩码
            image.save(os.path.join(output_dir, f"{image_name}_original.png"))
            pred_mask_pil.save(os.path.join(output_dir, f"{image_name}_pred_mask.png"))

            print(f"✅ 推理完成并保存: {image_name}")


if __name__ == "__main__":
    main()
    # model = deeplabv3_resnet50(weights='DEFAULT')
    # model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    # model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    # model = model.to(config.DEVICE)
    # test_dir = "/home/danzer/Documents/code/demo_shuzizhongguo/ForgeryAnalysis_Stage_1_Test/Image"
    # output_dir = "/home/danzer/Documents/code/demo_shuzizhongguo/inference_results"
    # val_transform = A.Compose([
    #         A.Resize(config.IMG_SIZE, config.IMG_SIZE),
    #         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #         ToTensorV2()
    #     ], is_check_shapes=False)
    # infer_and_save(model, test_dir, output_dir, 'deeplab_best_model.pth', val_transform)
