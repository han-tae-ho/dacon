import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from util import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset

WORK_PATH = './'

train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.3),
    A.Rotate(limit=[-20,20], p = 0.1),
    A.CLAHE(p=0.1),
    A.RandomBrightnessContrast(p=0.1),
    A.Normalize(),
    ToTensorV2()])

valid_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()])


class ObjectDataset(Dataset):
    def __init__(self, mode='train'): 
        self.mode = mode
        set_seed(2022)
        shuffled = np.random.permutation(50000)
        train_index, valid_index = shuffled[:45000], shuffled[45000:]

        df = pd.read_csv(WORK_PATH + 'train.csv')
        if self.mode == 'train':
            self.file_names = np.array(df['file_name'])[train_index]
            self.labels = np.array(df['class'][train_index])
        elif self.mode == 'valid':
            self.file_names = np.array(df['file_name'])[valid_index]
            self.labels = np.array(df['class'][valid_index])

        # self.file_names = np.array(self.csv['file_name'])
        # self.labels = np.array(self.csv['class'])
        
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image_path = WORK_PATH + 'train/' + self.file_names[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.mode == 'train': 
            image = train_transform(image = image)
        elif self.mode == 'valid':
            image = valid_transform(image = image)
        label = int(self.labels[idx])

        return image, label

class PesudoDataset(Dataset):
    def __init__(self): 
        df = pd.read_csv(WORK_PATH + 'pesudo.csv')
        self.file_names = np.array(df['id'])
        self.labels = np.array(df['target'])

        # self.file_names = np.array(self.csv['file_name'])
        # self.labels = np.array(self.csv['class'])
        
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image_path = WORK_PATH + 'test/' + self.file_names[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = valid_transform(image = image)
        label = int(self.labels[idx])

        return image, label


class TestDataset(Dataset):
    def __init__(self): 
        self.file_path = WORK_PATH + 'test/'
        self.sample_csv = pd.read_csv(WORK_PATH + 'sample_submission.csv')
        
    def __len__(self):
        return self.sample_csv.shape[0]

    def __getitem__(self, idx):
        folder_name = str(self.sample_csv.id[idx])
        image_path = os.path.join(self.file_path, folder_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = valid_transform(image = image)
        
        return image