import joblib
from catboost import CatBoostClassifier
from models import ConvNet
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset import HandTestDataset
from torch.utils.data import DataLoader

df_test = pd.read_csv('./data/test.csv')
X_test = df_test.drop(['id'], axis = 1)
BATCH_SIZE = 32
device = 'cpu'
print(device)

## lgb
model_lgb = joblib.load('./saved/best_lgb.pkl')
probs1 = model_lgb.predict(X_test)

## catboost
model_cat = CatBoostClassifier()
model_cat.load_model("./saved/best_catboost")
probs2 = model_cat.predict_proba(X_test)


## cnn model
cnn_model = ConvNet()
cnn_model.load_state_dict(torch.load('./saved/cnn_best_model.pt', map_location=device))
cnn_model.to(device)
test_loader = DataLoader(HandTestDataset(), batch_size=BATCH_SIZE, shuffle=False)

probs3 = []
with torch.no_grad():
    print("making test_ouput...")
    cnn_model.eval()
    for i, images in enumerate(tqdm(test_loader)):
        images = images.to(device)
        outputs = cnn_model(images)
        probs3.extend(outputs.tolist())

probs3 = np.array(probs3)


def soft_enesnble(weights):
    return probs1 * weights[0] + probs2 * weights[1] + probs3 * weights[2]


# ensemble
def eval():
    df = pd.read_csv('./data/sample_submission.csv')
    probs = soft_enesnble([0.3,0.3,0.4])
    df.target = [np.argmax(line) for line in probs]
    df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    eval()