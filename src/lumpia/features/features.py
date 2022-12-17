import pandas as pd

def gen_ohe(df: pd.DataFrame, cols: list):
    dummies_df = pd.get_dummies(df, columns=cols)
    return dummies_df