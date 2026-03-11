import torch
import torch.nn as nn



# --- 4. LOSS & IOU HELPER FUNCTIONS ---
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def combined_loss(pred, target,):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    # print(f"BCE Loss: {bce.item()}, Dice Loss: {dice.item()}, Combined Loss: {(bce + dice).item()}")
    return bce + dice

# NEW: IOU Calculation Function
def calculate_iou(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    # Convert probabilities to binary mask (0 or 1)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    return (intersection + smooth) / (union + smooth)

def classification_loss(pred, target):
    # 二分类用 BCEWithLogitsLoss，多分类用 CrossEntropyLoss
    return nn.BCEWithLogitsLoss()(pred, target)


if __name__ == "__main__":
    # 示例：使用 BCEWithLogitsLoss 进行二分类
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # 假设我们有一个批次的模型输出和对应的标签
    outputs = torch.tensor([[0.5], [-1.0], [2.0]])  # 模型输出（未经过 sigmoid）
    labels = torch.tensor([[1.0], [0.0], [1.0]])   # 真实标签
    
    loss = criterion(outputs, labels)
    print("Loss:", loss.item())