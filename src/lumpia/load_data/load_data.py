import pandas as pd

def read_data(path: str):
    '''
    Read database.
    
    Parameters: 
        path: path to data
        
    Returns:
        df (dataframe): database in pandas format
    '''
    df = pd.read_csv(path, index_col=0)
    return df
