import unittest

import numpy as np
from twlearn.metrics import Rmse, Mae

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
