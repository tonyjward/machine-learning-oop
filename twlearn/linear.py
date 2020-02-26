import numpy as np

class LinearRegression:
    def __init__(self):
        self.coef = None
        self.intercept = None

    def _add_bias(self, X):
        return np.c_[np.ones(X.shape[0]), X]

    def _ols(self, X, y):
        xTx = np.dot(X.T, X)
        inverse_xTx = np.linalg.inv(xTx)
        xTy = np.dot(X.T, y)
        bhat = np.dot(inverse_xTx, xTy)
        return bhat

    def fit(self, X, y, fit_intercept = True, optimiser = 'OLS'):
        """
        Fit model coefficients

        Arguments:
        X: 1D or 2D numpy array - X.shape = (no_examples, no_features)
        y: 1D numpy array y.shape = (no_examples, 1)
        """
        
        # Don't use rank 1 arrays 
        if len(X.shape) == 1:
            X = X.reshape(-1, 1) # https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape
        y = y.reshape(-1, 1)

        # Dimensions of problem        
        no_examples, no_features =  X.shape  

        # add bias if fit_intercept is True
        if fit_intercept:
            X = self._add_bias(X)

        # closed form solution
        if optimiser == 'OLS':
            bhat = self._ols(X, y)
   
        # set attributes
        if fit_intercept:
            self.intercept = bhat[0]
            self.coef = bhat[1:]
        else:
            self.intercept = bhat[0]
            self.coef = bhat

        assert(self.coef.shape == ((no_features, 1)))

    def coefficients(self):
        return {'intercept': self.intercept, 'coefficients': self.coef}

    def predict(self, X_new):
        if len(X_new.shape) == 1:
            X_new = X_new.reshape(-1, 1) 
        return self.intercept + np.dot(X_new, self.coef)