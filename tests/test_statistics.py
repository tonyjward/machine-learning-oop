import unittest

import numpy as np
from twlearn.statistics import five_by_two_cv, p_value

class Test_Five_By_Two_Cv(unittest.TestCase):
    def setUp(self):
        self.errorA = {0:{0:5., 1:4.},
                       1:{0:4., 1:3.},
                       2:{0:3., 1:3.},
                       3:{0:2., 1:3.},
                       4:{0:1., 1:3.}}
        self.errorB = {0:{0:7., 1:8.},
                       1:{0:6., 1:5.},
                       2:{0:3., 1:9.},
                       3:{0:6., 1:10.},
                       4:{0:8., 1:7.}}
    
    def test_five_by_two_cv(self):
        t_statistic, average_differences = five_by_two_cv(self.errorA, self.errorB)
        self.assertIsNone(np.testing.assert_allclose(t_statistic, -0.830454799))

class Test_P_Value_One_Sided(unittest.TestCase):
    """ Values tested against https://www.socscistatistics.com/pvalues/tdistribution.aspx"""
    def setUp(self):
        self.t_statistic = 1.5
        self.degrees_freedom = 5
        self.sided = 1
    
    def test_p_value(self):
        p = p_value(self.t_statistic, self.degrees_freedom, self.sided)
        self.assertIsNone(np.testing.assert_allclose(p, .096952, atol=1e-05))

class Test_P_Value_Two_Sided(unittest.TestCase):
    """ Values tested against https://www.socscistatistics.com/pvalues/tdistribution.aspx"""
    def setUp(self):
        self.t_statistic = 1.5
        self.degrees_freedom = 5
        self.sided = 2
    
    def test_p_value(self):
        p = p_value(self.t_statistic, self.degrees_freedom, self.sided)
        self.assertIsNone(np.testing.assert_allclose(p, 0.193904, atol=1e-05))

if __name__ == '__main__':
    unittest.main()