from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
import numpy as np
import pandas as pd
from utils import add_aug
import lightgbm as lgb
import joblib
from catboost import Pool, CatBoostClassifier
from models import ConvNet
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataset import HandDataset
import torch
import torch.nn.functional as F
import torch.nn as nn


df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')

X = df_train.drop(['id', 'target'], axis = 1)
y = df_train.target

# Split k-fold validation
n_splits = 5
random_state = np.random.seed(2022)

skf = KFold(n_splits=n_splits, random_state=random_state, shuffle=False)

# lgbm train
def lgbm_train():
    params = {}
    params['boosting_type']='gbdt' #GradientBoostingDecisionTree
    params['objective']='multiclass' #Multi-class target feature
    params['metric']='multi_error' #metric for multi-class
    params['num_class']=4
    params['force_col_wise']='true'
    params['verbose']=-1

    best_acc = 0
    acc = []
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.loc[train_idx], X.loc[val_idx]
        y_train, y_val = y.loc[train_idx], y.loc[val_idx]

        # Data augmentation
        X_train, y_train = add_aug(X_train, y_train, n=1)

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        # Train the model
        model_lgb = lgb.train(params,
                        train_data,              
                        num_boost_round = 1000,
                        valid_sets=val_data,
                        verbose_eval=100,
                        early_stopping_rounds=100) 

        # Out of fold vector
        y_preds = model_lgb.predict(X_val)
        y_preds = [np.argmax(line) for line in y_preds]
        score = precision_score(y_preds, y_val,average=None).mean()
        print(score)
        acc.append(score)

        if score > best_acc:
            best_acc = score
            joblib.dump(model_lgb, './saved/best_lgb.pkl')


# train catboost
def catboost_train():
    best_acc = 0
    acc = []
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.loc[train_idx], X.loc[val_idx]
        y_train, y_val = y.loc[train_idx], y.loc[val_idx]

        X_train, y_train = add_aug(X_train, y_train, n=1)

        train_dataset = Pool(data=X_train,
                            label=y_train)

        eval_dataset = Pool(data=X_val,
                            label=y_val)

        model_cat = CatBoostClassifier(depth=3,
                            iterations=1000,
                            loss_function='MultiClass',
                            random_seed=23,
                            early_stopping_rounds=50,
                            verbose = 100,
                            # task_type="GPU"
                            )

        model_cat.fit(train_dataset, eval_set=(X_val, y_val), use_best_model=True)

        y_preds = model_cat.predict(eval_dataset)
        score = precision_score(y_preds, y_val,average=None).mean()
        print(score)
        acc.append(score)

        if score > best_acc:
            best_acc = score
            model_cat.save_model("./saved/best_catboost")

    print(np.mean(acc)) 


## CNN
best_val_loss = 1e9
best_val_acc = 0
device = 'cuda'
EPOCH = 20
BATCH_SIZE = 32
MODEL_PATH = './saved'

def cnn_train():
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        model = ConvNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience = 5, verbose = 1)
        
        train_loader = DataLoader(HandDataset(train_idx), batch_size=BATCH_SIZE, shuffle= True)
        valid_loader = DataLoader(HandDataset(val_idx), batch_size=BATCH_SIZE, shuffle= False)

        model.to(device)
        for epoch in range(EPOCH):
            model.train()
            loss_value = 0
            matches = 0
            

            for idx, train_batch in enumerate(tqdm(train_loader)):
                inputs, labels = train_batch
                input = inputs.to(device)
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

                    # print(
                    #     f"Epoch[{epoch+1}/{EPOCH}]({idx + 1}/{len(train_loader)}) || "
                    #     f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%}"
                    # )

            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                val_size = 0
                for i, (images, labels) in enumerate(tqdm(valid_loader)):
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=-1)

                    loss_item = criterion(outputs, labels).item()

                    acc_item = (labels == preds).sum().item()
                    val_size += len(labels)
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                val_loss = np.sum(val_loss_items) / len(valid_loader)
                val_acc = np.sum(val_acc_items) / val_size

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                if val_acc > best_val_acc:
                    print("New best model for val accuracy! saving the model..")
                    torch.save(model.state_dict(), f"./saved/cnn_best_model.pt")
                    best_val_acc = val_acc
                
                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                
            scheduler.step(val_loss)


if __name__ == '__main__':
    lgbm_train()
    catboost_train()
    cnn_train()