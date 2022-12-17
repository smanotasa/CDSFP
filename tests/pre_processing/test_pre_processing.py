import unittest
import pytest
from pandas.testing import assert_frame_equal
from lumpia.pre_processing.pre_processing import drop_nan
from lumpia.pre_processing.pre_processing import fill_mean
import pandas as pd

class Testdatacleanup(unittest.TestCase):
    
    def test_drop_nan(self):
        column = ['col'] 
        df = pd.DataFrame({'col': [0,5,10, np.nan]})
        output = drop_nan(df, column)
        expected_output = pd.DataFrame({'col': [0,5,10]})
        assert_frame_equal(output,expected_output,check_dtype=False)
        
    def test_fill_mean(self):
        df = pd.DataFrame({'col': [0,5,10, np.nan]})
        column = ['col'] 
        output = fill_mean(df,column)
        expected_output = pd.DataFrame({'col': [0,5,10,5]})
        assert_frame_equal(output,expected_output,check_dtype=False)
