import unittest

import numpy as np
from twlearn.metrics import Rmse, Mae, five_by_two_cv

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

class Test_1_solution(unittest.TestCase):
    def setUp(self):
        self.predictions = np.array([1, 2, 3])
        self.actuals = np.array([0.9, 2.2, 2.7])

    def test_Rmse(self):
        rmse = Rmse(self.predictions, self.actuals)
        self.assertIsNone(np.testing.assert_allclose(rmse, 0.374165739))

    def test_Mae(self):
        mae = Mae(self.predictions, self.actuals)
        self.assertIsNone(np.testing.assert_allclose(mae, 0.2))

class Test_2_solutions(unittest.TestCase):
    def setUp(self):
        self.predictions = np.array([[1,1], [2, 2], [3, 3]])
        self.actuals = np.array([0.9, 2.2, 2.7])

    def test_Rmse(self):
        rmse = Rmse(self.predictions, self.actuals)
        self.assertIsNone(np.testing.assert_allclose(rmse, np.array([[0.374165739, 0.374165739]])))

    def test_Mae(self):
        mae = Mae(self.predictions, self.actuals)
        self.assertIsNone(np.testing.assert_allclose(mae, np.array([[0.2, 0.2]])))

if __name__ == '__main__':
    unittest.main()
