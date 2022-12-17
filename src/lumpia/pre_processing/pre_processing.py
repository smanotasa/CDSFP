import numpy as np

class preprocessor:
  def __init__(self,df,column):
        self.df = df
        self.column = list(column)

  def drop_nan(self):
       for i in self.column:
            self.df = self.df.dropna(subset=[i])
       return self.df

  def fill_mean(self):
      self.df[self.column] = self.df[self.column].fillna(np.mean(self.df[self.column]))
      return self.df
