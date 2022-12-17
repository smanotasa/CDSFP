import pandas as pd
from sklearn.model_selection import train_test_split

def split(df: pd.DataFrame):
    '''
    Split data between train and test using sklearn.model_selection.train_test_split.
    
    Parameters: 
        df (dataframe): dataframe to split data for
        
    Returns:
        X_train (dataframe): splitted X train dataframe from train_test_split
        X_test (dataframe): splitted X test dataframe from train_test_split
        y_train (dataframe): splitted y train dataframe from train_test_split
        y_test (dataframe): splitted y test dataframe from train_test_split
        
    '''
    
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2, stratify=y)
    
    return X_train, X_test, y_train, y_test
