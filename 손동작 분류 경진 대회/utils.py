import numpy as np
import pandas as pd

def add_aug(df_train, df_valid, n = 1):
    df_aug = df_train.copy()
    dfy = df_valid.copy()
    for _ in range(n):
        np.random.seed(_ * 100)
        df2 = pd.DataFrame([np.random.uniform(0.9, 1.1, df_train.shape[1]) for i in range(df_train.shape[0])],
                    columns=df_train.columns)
        df_temp = df_train.reset_index().drop('index', axis=1).mul(df2)
        df_aug = pd.concat([df_aug, df_temp])
        dfy = pd.concat([dfy, df_valid])

    return df_aug, dfy