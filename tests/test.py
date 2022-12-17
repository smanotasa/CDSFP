import unittest
import pytest
from pandas.testing import assert_frame_equal
#%%
from src.lumpia import *
#%%
import pandas as pd
from datetime import datetime
class tests(unittest.TestCase):

# tests for preprocessing
def test_read_data(self):
        ans = pd.read_csv("churn_data.csv")
        res = read_data("churn_data.csv")
        assert_frame_equal(res, ans)
        
# todo: tests for feature computation
