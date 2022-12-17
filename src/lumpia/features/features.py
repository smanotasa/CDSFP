import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def gen_dummies(df: pd.DataFrame, cols: list):
    dummies_df = pd.get_dummies(df, columns=cols)
    return dummies_df

def normalize(df: pd.DataFrame, cols: list):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled, columns = cols, index = df.index)
    return scaled_df

def take_log(df: pd.DataFrame, cols: list):
    df[cols] = df[cols].apply(np.log)
    return df
