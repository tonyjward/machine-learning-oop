import numpy as numpy

class MyLinearRegression:
    def __init__(self, fit_intercept = True):
        self.coef_ = None
        self.intercept_ = None
        self._fit_intercept = fit_intercept

    def fit(self, X, y):

        # TODO: implemented optimisation algorithm
        
        self.coef = None
        self.intercept = None

        return coef, intercept

    def predict(self, X_new):

        predictions = None
        return predictions