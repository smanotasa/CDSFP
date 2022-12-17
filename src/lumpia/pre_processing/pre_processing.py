import numpy as np
import pandas as pd

def drop_nan(df: pd.DataFrame, cols: list):

    cleaned_df = df.dropna(axis=0, subset=cols, how='any')
    return cleaned_df

def fill_mean(df: pd.DataFrame, cols: list):

    df[cols] = df[cols].fillna(df.mean().iloc[0])
    return df 