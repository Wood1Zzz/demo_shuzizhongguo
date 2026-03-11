import torch

# Configuration file for training parameters
CSV_PATH = '/home/danzer/Documents/code/demo_shuzizhongguo/ForgeryDataset.csv'
BATCH_SIZE = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 10
LEARNING_RATE = 1e-4
IMG_SIZE = 512
MODEL_PTH = 'deeplab_best_model.pth'