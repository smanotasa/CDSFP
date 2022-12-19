import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def gen_dummies(df: pd.DataFrame, cols: list):
    dummies_df = pd.get_dummies(df, columns=cols, drop_first=True)
    return dummies_df

def normalize(df: pd.DataFrame, cols: list):
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


def take_log(df: pd.DataFrame, cols: list):
    df.loc[:, cols] = np.where(df[cols]>0, np.log(df[cols]), df[cols])
    return df
