import numpy as np

class LinearRegression:
    def __init__(self, fit_intercept = True):
        self.coef = None
        self.intercept = None
        self._fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit model coefficients

        Arguments:
        X: 1D or 2D numpy array - X.shape = (m, n)
        y: 1D numpy array y.shape = (n, 1)
        """

        # Don't use rank 1 arrays 
        if len(X.shape) == 1:
            X = X.reshape(-1, 1) # https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape
        y = y.reshape(-1, 1)

        # Dimensions of problem        
        no_examples, no_features =  X.shape  

        # add bias if fit_intercept is True
        if self._fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # closed form solution
        xTx = np.dot(X.T, X)
        inverse_xTx = np.linalg.inv(xTx)
        xTy = np.dot(X.T, y)
        bhat = np.dot(inverse_xTx, xTy)
   
        # set attributes
        if self._fit_intercept:
            self.intercept = bhat[0]
            self.coef = bhat[1:]
        else:
            self.intercept_ = bhat[0]
            self.coef = bhat

        assert(self.coef.shape == ((no_features, 1)))

    def coefficients(self):
        return {'intercept': self.intercept, 'coefficients': self.coef}


    def predict(self, X_new):

        predictions = np.dot(X_new, self.coef_)
        return predictions