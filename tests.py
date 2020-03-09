import unittest

import numpy as np
from twlearn import LinearRegression, generate_dataset
from twlearn.metrics import Rmse, Mae, five_by_two_cv

class Test_Linear_Regression_OLS_1D_Class(unittest.TestCase):
    
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

class Test_LinearRegression_OLS_2D_MSE_Class(unittest.TestCase):
    
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

class Test_LinearRegression_GradientDescent_MSE_2D_Class(unittest.TestCase):
    
    def setUp(self):
        X, y =  np.array([[1.,0],[0.,1.],[0.,0.]]), np.array([[0.],[1.],[2.]])
        self.lm = LinearRegression()
        self.lm.fit(X, y, optimiser = 'GD', loss = 'MSE', debug = False)

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

class Test_LinearRegression_ParticleSwarm_MAE_1D_Class(unittest.TestCase):
    
    def setUp(self):
        X, y = np.array([[-1.], [0.], [1.]]), np.array([[0.],[1.],[2.]])
        self.lm = LinearRegression()
        self.lm.fit(X, y, optimiser = 'PSO', loss = Mae, upper = 4, lower = -4)

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

# class TestLinearRegressionGradientDescent_MAE_1D_Class(unittest.TestCase):
    
#     def setUp(self):
#         X, y = np.array([[-1.], [0.], [1.]]), np.array([[0.],[1.],[2.]])
#         self.lm = LinearRegression()
#         self.lm.fit(X, y, optimiser = 'GD', loss = 'MAE', learning_rate = 0.001, num_iterations = 1000, debug = True)

#     def test_fit_1D_with_intercept(self):
#         desired_intercept, desired_coefficients = np.array([1.]), np.array([[1.]])    
#         coefficients = self.lm.coefficients()
#         self.assertIsNone(np.testing.assert_allclose(desired_intercept, coefficients['intercept']))
#         self.assertIsNone(np.testing.assert_allclose(desired_coefficients, coefficients['coefficients']))

#     def test_predict_1D_with_intercept(self):
#         X_new = np.array([[-0.5], [2.0], [1.5]])
#         desired_yhat = np.array([[0.5],[3.],[2.5]])
#         yhat = self.lm.predict(X_new)
#         self.assertIsNone(np.testing.assert_allclose(desired_yhat, yhat))

if __name__ == '__main__':
    unittest.main()
