from torch.utils.data import Dataset
from utils import add_aug
import pandas as pd
import numpy as np
import torch

class HandDataset(Dataset):
    def __init__(self, index): 
        self.index = index.tolist()

        df_train = pd.read_csv('./data/train.csv')
        X = df_train.drop(['id', 'target'], axis = 1)
        y = df_train.target

        X,y = add_aug(X, y, n = 1)
        X=(X-X.mean())/X.std()
        
        if self.index:
            self.images = X.loc[self.index]
            self.labels = y.loc[self.index]
        else:
            self.images = X
            self.labels = y
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images.iloc[idx]
        image = np.array(image).reshape(-1, 8, 4)
        image = torch.Tensor(image)
        label = self.labels.iloc[idx]

        return image, label

class HandTestDataset(Dataset):
    def __init__(self): 
        df_train = pd.read_csv('./data/test.csv')
        X = df_train.drop(['id'], axis = 1)
        X=(X-X.mean())/X.std()
        
        self.images = X
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images.iloc[idx]
        image = np.array(image).reshape(-1, 8, 4)
        image = torch.Tensor(image)

        return image