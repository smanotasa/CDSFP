import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def gen_dummies(df: pd.DataFrame, cols: list):
    dummies_df = pd.get_dummies(df, columns=cols, drop_first=True)
    return dummies_df

def normalize(df: pd.DataFrame, cols: list):
    df_toscale = df.loc[:, df.columns.isin(cols)]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_toscale)
    scaled_df = pd.DataFrame(scaled, columns=cols)
    df_appended = pd.concat([scaled_df, df.loc[:, ~df.columns.isin(cols)]], axis=1)
    return df_appended


def take_log(df: pd.DataFrame, cols: list):
    df.loc[:, cols] = np.where(df[cols]>0, np.log(df[cols]), df[cols])
    return df
