import pandas as pd
from sklearn.model_selection import train_test_split

class dataset:
    
    def __init__(self, file):
        self.file = file
        self.loadnsplitter(self.file)
            
    def loadnsplitter(self, file):
        try:
            self.df = pd.read_csv(file, index_col=0, decimal='.')
        except NameError:
            print('There is no such file')
        except FileNotFoundError:
            print('There is no such file')
            
        self.train, self.test = train_test_split(self.df, random_state = 420, train_size = .80)
        return self.train, self.test