import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import timm 



EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
IMG_SIZE = 224


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
    dataset = ForgeryClassDataset('ForgeryDataset.csv')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    model = ForgeryClassifier()
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(10):
        model.train()
        total_loss = 0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}')


if __name__ == "__main__":
    main()