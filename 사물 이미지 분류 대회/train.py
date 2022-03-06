from model import SRCNN_EFB0
from dataset import *
import os
import torch.nn as nn
import torch
import numpy as np
import tqdm
from torch.utils.data import DataLoader

def train():
    ############## config ################################################################################
    model = SRCNN_EFB0()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WORK_PATH = './'
    MODEL_PATH = './saved'
    LEARNING_RATE = 0.002
    EPOCH = 40
    BATCH_SIZE = 64

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience = 2, verbose = 1)

    trainloader = DataLoader(ObjectDataset('train'), batch_size=BATCH_SIZE, shuffle = True)
    validloader = DataLoader(ObjectDataset('valid'), batch_size=BATCH_SIZE, shuffle = True)
    testloader = DataLoader(TestDataset(), batch_size = BATCH_SIZE, shuffle=False, drop_last=False)

#######################################################################################################

    model.to(device)
    best_val_loss = 1e9
    best_val_acc = 0
    for epoch in range(EPOCH):
        model.train()
        loss_value = 0
        matches = 0
        

        for idx, train_batch in enumerate(tqdm(trainloader)):
            inputs, labels = train_batch
            input = inputs['image'].to(device)
            label = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input)
            
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=-1)
            loss_value += loss.item()
            matches += (preds == label).sum().item()
            acc = (outputs.argmax(dim=1) == label).float().mean()

            if (idx + 1) % 100 == 0:
                train_loss = loss_value / (idx+1)
                train_acc = matches / BATCH_SIZE / (idx+1)

                print(
                    f"Epoch[{epoch+1}/{EPOCH}]({idx + 1}/{len(trainloader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%}"
                )

        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            val_size = 0
            for i, (images, labels) in enumerate(tqdm(validloader)):
                images = images['image'].to(device)
                labels = labels.to(device)

                outputs = model(images)
                preds = torch.argmax(outputs, dim=-1)

                loss_item = criterion(outputs, labels).item()

                acc_item = (labels == preds).sum().item()
                val_size += len(labels)
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

            val_loss = np.sum(val_loss_items) / len(validloader)
            val_acc = np.sum(val_acc_items) / val_size

            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_acc > best_val_acc:
                print("New best model for val accuracy! saving the model..")
                torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"efbo_epoch{epoch}.pt"))
                best_val_acc = val_acc
            
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
        scheduler.step(val_loss)

        ##### pesudo
        if best_val_acc > 0.8:
            model.eval()
            pred_list = []
            confidences = []
            with torch.no_grad():
                print("making pesudo labeing...")
                for i, images in enumerate(tqdm(testloader)):
                    images = images['image'].to(device)
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=-1)
                    pred_list.extend(preds.tolist())
                    confidence = outputs.max(dim=1)[0] > 10
                    confidences.extend(confidence.tolist())

            sample_csv = pd.read_csv(WORK_PATH + '/sample_submission.csv')
            sample_csv.target = pred_list
            sample_csv['confidence'] = confidences
            sample_csv.to_csv(WORK_PATH + '/pesudo.csv', index = False)


            pesudoloader = torch.utils.data.DataLoader(PesudoDataset(), batch_size=BATCH_SIZE, shuffle = True)
            print("train pesudo labeing...")
            for idx, train_batch in enumerate(tqdm(pesudoloader)):
                inputs, labels = train_batch
                input = inputs['image'].to(device)
                label = labels.to(device)

                optimizer.zero_grad()
                outputs = model(input)
                
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()


if __name__ == "__main__":
    train()