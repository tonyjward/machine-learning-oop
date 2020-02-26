import unittest

import numpy as np
from twlearn import LinearRegression, generate_dataset
from twlearn.metrics import Rmse, Mae

class TestLinearRegressionClass(unittest.TestCase):
    def test_fit_1D_with_intercept(self):
        X , y = np.array([[-1.], [0.], [1.]]), np.array([[0.],[1.],[2.]])
        lm = LinearRegression()
        lm.fit(X, y)
        coefficients = lm.coefficients()
        self.assertEqual(coefficients['intercept'], 1)
        self.assertEqual(coefficients['coefficients'], 1)

    def test_fit_2D_with_intercept(self):
        X , y =  np.array([[1.,0],[0.,1.],[0.,0.]]), np.array([[0.],[1.],[2.]])
        desired_intercept, desired_coefficients = np.array([2.]), np.array([[-2.], [-1.]])
        lm = LinearRegression()
        lm.fit(X, y)
        coefficients = lm.coefficients()
        np.testing.assert_allclose(desired_intercept, coefficients['intercept'])
        np.testing.assert_allclose(desired_coefficients, coefficients['coefficients'])
                
if __name__ == '__main__':
    unittest.main()
