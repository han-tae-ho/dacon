import ttach as tta
import pandas as pd
from model import SRCNN_EFB0
from dataset import *
import os
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

def eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WORK_PATH = './'
    MODEL_PATH = './saved/'
    BATCH_SIZE = 64

    model = SRCNN_EFB0()
    model.load_state_dict(torch.load(MODEL_PATH + 'efbo_epoch40.pt',  map_location= device))
    model.to(device)

    testloader = DataLoader(TestDataset(), batch_size = BATCH_SIZE, shuffle=False, drop_last=False)

    print(len(testloader))
    tta_transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Multiply(factors=[0.8,0.85,0.9,0.95, 1, 1.05, 1.1, 1.15, 1.2]),  
        ]
    )
    tta_model = tta.ClassificationTTAWrapper(model, tta_transforms)

    model.eval()
    pred_list = []
    with torch.no_grad():
        print("making test_ouput...")
        model.eval()
        for i, images in enumerate(tqdm(testloader)):
            images = images['image'].to(device)
            # outputs = model(images)
            # preds1 = torch.argmax(outputs, dim=-1)

            outputs = tta_model(images)
            preds = torch.argmax(outputs, dim=-1)
            pred_list.extend(preds.tolist())

    class_dict = dict(zip([i for i in range(10)], os.listdir(WORK_PATH + 'train')))
    classes = [class_dict[x] for x in pred_list]
    sample_csv = pd.read_csv(WORK_PATH + 'sample_submission.csv')
    sample_csv.target = classes
    sample_csv.to_csv(WORK_PATH + 'submission.csv', index = False)

if __name__ == "__main__":
    eval()