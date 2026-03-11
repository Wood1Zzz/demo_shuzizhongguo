import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

class DeeplabWithClassification(nn.Module):
    def __init__(self, num_classes=1, num_classes_seg=2, num_classes_cls=2):
        super(DeeplabWithClassification, self).__init__()
        self.deeplab = deeplabv3_resnet50(weights='DEFAULT', num_classes=num_classes_seg)
        self.deeplab.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        self.deeplab.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        # self.classifier = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        #     nn.Linear(256, num_classes_cls)
        # )

    def forward(self, x):
        seg_out = self.deeplab(x)
        # 获取 backbone 最后特征
        features = self.deeplab.backbone(x)['out']
        # cls_out = self.classifier(features)
        return seg_out





if __name__ == "__main__":
    # print("Loading DeepLabV3 with ImageNet Weights...")
    # try:
    #     model = deeplabv3_resnet50(weights='DEFAULT')
    #     print("✅ Loaded ImageNet Weights!")
    # except:
    #     model = deeplabv3_resnet50(weights=None)

    # model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    # model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

    # model = model.to(config.DEVICE)
    # optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    pass