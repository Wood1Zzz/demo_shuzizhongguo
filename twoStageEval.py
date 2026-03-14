import torch
import os
from PIL import Image
from torchvision import transforms
import torchvision
import glob
import config
from model import deeplabv3_resnet50
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import torch.nn as nn
import csv
import numpy as np




class ForgeryClassifier(nn.Module):
    def __init__(self):
        super(ForgeryClassifier, self).__init__()
        # self.model = timm.create_model('resnet50', pretrained=False)
        self.model = torchvision.models.resnet50(pretrained=False)
        # 替换最后一层为输出1个logit
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)



def infer_two_stage_model():
    """
    加载训练好的模型，对测试目录中的图片进行推理，并保存预测结果。
    """
    test_dir = '/root/autodl-tmp/demo_shuzizhongguo/ForgeryAnalysis_Stage_1_Test'
    output_dir = '/root/autodl-tmp/demo_shuzizhongguo/'
    real_dir = os.path.join(output_dir, 'twoStageResult', 'real')
    fake_dir = os.path.join(output_dir, 'twoStageResult', 'fake')
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # 定义测试图片的预处理变换
    val_cls_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整为分类模型的输入尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    model_classification = ForgeryClassifier()
    model_classification = model_classification.to(config.DEVICE)
    model_classification.load_state_dict(torch.load('./class_best_model.pth', map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    model_classification.eval()

    img_paths = glob.glob(os.path.join(test_dir, 'Image', '*.*'))
    cls_csv_path = os.path.join(output_dir, 'classification_results.csv')
    with open(cls_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Image Name', 'Predicted Label'])  # 写入表头

    for img_path in img_paths:
        image = Image.open(img_path).convert('RGB')
        image_tensor = val_cls_transform(image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            output = model_classification(image_tensor)
            pred_label = (torch.sigmoid(output) > 0.5).float().item()
        # 根据预测标签将图片保存到对应的目录
        if pred_label == 1:
            img_path = os.path.join(test_dir, os.path.basename(img_path))
            label = 0
            image.save(os.path.join(real_dir, os.path.basename(img_path)))
        else:
            img_path = os.path.join(test_dir, os.path.basename(img_path))
            label = 1
            image.save(os.path.join(fake_dir, os.path.basename(img_path)))
        with open(cls_csv_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([os.path.basename(img_path), label])  # 写入图片名称和预测标签
    print("Finish total inference for image!!")
    
    # 加载分割模型并进行推理
    print("Start inference for segmentation on predicted fake images...")
    model_segmentation = deeplabv3_resnet50(weights='DEFAULT')
    model_segmentation.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    model_segmentation.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    model_segmentation = model_segmentation.to(config.DEVICE)
    model_segmentation.classifier[4] = nn.Sequential(
        nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
        nn.ReLU(),
        nn.Dropout(p=0.5),  # 加入 Dropout
        nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    )

    model_segmentation.aux_classifier[4] = nn.Sequential(
        nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
        nn.ReLU(),
        nn.Dropout(p=0.5),  # 加入 Dropout
        nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    )
    model_segmentation.load_state_dict(torch.load('./deeplab_seg_best_model.pth', map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    model_segmentation = model_segmentation.to(config.DEVICE)
    model_segmentation.eval()

    val_seg_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], is_check_shapes=False)

    test_csv = os.path.join(output_dir, 'classification_results.csv')
    print("Start inference for segmentation on predicted fake images!!")

    seg_infer_dir = os.path.join(output_dir, 'twoStageResult', 'infer')
    os.makedirs(seg_infer_dir, exist_ok=True)
    
    with open(test_csv, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            pred_label = int(row['Predicted Label'])
            if pred_label == 1:  # 伪造图像
                img_name = row['Image Name']
                print("✅ Processing image:", img_name)
                img_path = os.path.join(test_dir, 'Image', img_name)
                image = Image.open(img_path).convert('RGB')
                h, w = image.size
                image_tensor = val_seg_transform(image=np.array(image))['image'].unsqueeze(0).to(config.DEVICE)
                output = model_segmentation(image_tensor)['out']
                pred_mask = (torch.sigmoid(output[0]) > 0.5).float().cpu()
                pred_mask = F.interpolate(pred_mask.unsqueeze(0), size=(w, h), mode='nearest').squeeze(0)
                pred_mask_pil = transforms.ToPILImage()(pred_mask.squeeze(0))
                # pred_mask_pil.save(os.path.join(fake_dir, f"{os.path.splitext(img_name)[0]}_pred_mask.png"))

                # 保存原始图片和预测掩码
                image.save(os.path.join(output_dir, 'twoStageResult', 'infer', f"{img_name}_original.png"))
                pred_mask_pil.save(os.path.join(output_dir, 'twoStageResult', 'infer', f"{img_name}_pred_mask.png"))

                # mask覆盖到原图（红色区域）
                image_np = np.array(image)
                mask_np = np.array(pred_mask_pil)
                mask_color = np.zeros_like(image_np)
                mask_color[..., 0] = mask_np  # 红色通道
                mask_color[..., 1] = 0
                mask_color[..., 2] = 0

                alpha = 0.4  # 透明度
                overlay = image_np * (1 - alpha) + mask_color * alpha
                overlay = overlay.astype(np.uint8)
                overlay_img = Image.fromarray(overlay)
                overlay_img.save(os.path.join(output_dir, 'twoStageResult', 'infer', f"{img_name}_overlay.png"))

                print(f"✅ 推理完成并保存: {img_name}")

if __name__ == "__main__":
    infer_two_stage_model()