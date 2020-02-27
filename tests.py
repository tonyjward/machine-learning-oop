import unittest

import numpy as np
from twlearn import LinearRegression, generate_dataset
from twlearn.metrics import Rmse, Mae

class TestLinearRegressionClass1D(unittest.TestCase):
    
    def setUp(self):
        X, y = np.array([[-1.], [0.], [1.]]), np.array([[0.],[1.],[2.]])
        self.lm = LinearRegression()
        self.lm.fit(X, y)

    def test_fit_1D_with_intercept(self):
        desired_intercept, desired_coefficients = np.array([1.]), np.array([[1.]])    
        coefficients = self.lm.coefficients()
        self.assertIsNone(np.testing.assert_allclose(desired_intercept, coefficients['intercept']))
        self.assertIsNone(np.testing.assert_allclose(desired_coefficients, coefficients['coefficients']))

    def test_predict_1D_with_intercept(self):
        X_new = np.array([[-0.5], [2.0], [1.5]])
        desired_yhat = np.array([[0.5],[3.],[2.5]])
        yhat = self.lm.predict(X_new)
        self.assertIsNone(np.testing.assert_allclose(desired_yhat, yhat))

class TestLinearRegressionClass2D(unittest.TestCase):
    
    def setUp(self):
        X, y =  np.array([[1.,0],[0.,1.],[0.,0.]]), np.array([[0.],[1.],[2.]])
        self.lm = LinearRegression()
        self.lm.fit(X, y)

    def test_fit_2D_with_intercept(self):
        desired_intercept, desired_coefficients = np.array([2.]), np.array([[-2.], [-1.]])    
        coefficients = self.lm.coefficients()
        self.assertIsNone(np.testing.assert_allclose(desired_intercept, coefficients['intercept']))
        self.assertIsNone(np.testing.assert_allclose(desired_coefficients, coefficients['coefficients']))

    def test_predict_2D_with_intercept(self):
        X_new = np.array([[-0.5, 0], [2.0, 1], [0, 1.5]])
        desired_yhat = np.array([[3.],[-3.0],[0.5]])
        yhat = self.lm.predict(X_new)
        self.assertIsNone(np.testing.assert_allclose(desired_yhat, yhat))

class TestLinearRegressionClass2DGradientDescent(unittest.TestCase):
    
    def setUp(self):
        X, y =  np.array([[1.,0],[0.,1.],[0.,0.]]), np.array([[0.],[1.],[2.]])
        self.lm = LinearRegression()
        self.lm.fit(X, y, optimiser = 'GD')

    def test_fit_2D_with_intercept(self):
        desired_intercept, desired_coefficients = np.array([2.]), np.array([[-2.], [-1.]])    
        coefficients = self.lm.coefficients()
        self.assertIsNone(np.testing.assert_allclose(desired_intercept, coefficients['intercept']))
        self.assertIsNone(np.testing.assert_allclose(desired_coefficients, coefficients['coefficients']))

    def test_predict_2D_with_intercept(self):
        X_new = np.array([[-0.5, 0], [2.0, 1], [0, 1.5]])
        desired_yhat = np.array([[3.],[-3.0],[0.5]])
        yhat = self.lm.predict(X_new)
        self.assertIsNone(np.testing.assert_allclose(desired_yhat, yhat))

if __name__ == '__main__':
    unittest.main()
