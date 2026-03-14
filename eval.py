import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
import config
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import cv2
import glob
from PIL import Image
from torchvision import transforms
import numpy as np
import torch.nn.functional as F



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

            # # 推理尺寸与输入一致
            output = model(image_tensor)['out']
            pred_mask = (torch.sigmoid(output[0]) > 0.5).float().cpu()
            pred_mask = F.interpolate(pred_mask.unsqueeze(0), size=(w, h), mode='nearest').squeeze(0)
            
            # # 推理尺寸与输入不一致
            # output = model(image_tensor)['out']
            # pred_mask = (torch.sigmoid(output[0])).float().cpu()
            # pred_mask = (pred_mask > 0.5).float()  # 二值化掩码
            
            # 转换为PIL图像
            pred_mask_pil = to_pil(pred_mask.squeeze(0))  # 修复：移除多余的维度
            image_name = os.path.basename(image_path)

            # 保存原始图片和预测掩码
            image.save(os.path.join(output_dir, f"{image_name}_original.png"))
            pred_mask_pil.save(os.path.join(output_dir, f"{image_name}_pred_mask.png"))

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
            overlay_img.save(os.path.join(output_dir, f"{image_name}_overlay.png"))

            print(f"✅ 推理完成并保存: {image_name}")


if __name__ == "__main__":
    model = deeplabv3_resnet50(weights='DEFAULT')
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    model = model.to(config.DEVICE)
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
    test_dir = "/root/autodl-tmp/demo_shuzizhongguo/ForgeryAnalysis_Stage_1_Test/Image"
    output_dir = "/root/autodl-tmp/demo_shuzizhongguo/inference_results_overlay"
    val_transform = A.Compose([
            A.Resize(config.IMG_SIZE, config.IMG_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], is_check_shapes=False)
    infer_and_save(model, test_dir, output_dir, 'deeplab_seg_best_model.pth', val_transform)