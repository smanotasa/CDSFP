
from lumpia.features.features import gen_dummies
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

class Testfeatures(unittest.TestCase):
    
    def test_gen_dummy(self):
        data = pd.DataFrame({'col': ['Caucasian','Hispanic','Caucasian','Asian']})
        column = ['col'] 
        output = gen_dummies(data,column)
        expected_output = pd.DataFrame({'col_Asian': [0,0,0,1],'col_Caucasian': [1,0,1,0], 'col_Hispanic': [0,1,0,0]})
        assert_frame_equal(output,expected_output,check_dtype=False)