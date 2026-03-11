import torch

# Configuration file for training parameters
DATASET_PATH = '/home/danzer/Documents/code/demo_shuzizhongguo/ForgeryAnalysis_Stage_1_Train'
BATCH_SIZE = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 15
LEARNING_RATE = 1e-4
IMG_SIZE = 512
MODEL_PTH = 'deeplab_best_model.pth'