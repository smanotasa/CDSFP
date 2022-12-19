
from lumpia.features.features import gen_dummies
from lumpia.features.features import normalize
from lumpia.features.features import take_log
import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

class Testfeatures(unittest.TestCase):
    
    def test_gen_dummies(self):
        data = pd.DataFrame({'col': ['Caucasian','Hispanic','Caucasian','Caucasian']})
        column = ['col'] 
        output = gen_dummies(data,column)
        expected_output = pd.DataFrame({'col_Hispanic': [0,1,0,0]})
        assert_frame_equal(output,expected_output,check_dtype=False)
    
    def test_normalize(self):
        data = pd.DataFrame({'Hours': [1,2,3], 'Mins': [20,10,50]})
        column = ['Hours','Mins']
        output = normalize(data,column)
        expected_output = pd.DataFrame(
            {'Hours': [-1.224744871391589, 0.000000, 1.224744871391589],
             'Mins': [-0.3922322702763682, -0.9805806756909203, 1.3728129459672884]}
        )
        assert_frame_equal(output,expected_output,check_dtype=False)

    def test_take_log(self):
        data = pd.DataFrame({'Hours': [1,2,3], 'Mins': [20,10,50]})
        column = ['Hours','Mins']
        output = take_log(data,column)
        expected_output = pd.DataFrame({'Hours': [0.000000,0.693147,1.098612], 'Mins': [2.995732,2.302585,3.912023]})
        assert_frame_equal(output,expected_output,check_dtype=False)
