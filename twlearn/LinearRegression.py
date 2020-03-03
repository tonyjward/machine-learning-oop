import numpy as np
from math import floor
from .GeneralisedLinearModel import Glm
from .GradientDescent import GradDesc
from .ParticleSwarm import Pso

class LinearRegression(Glm, GradDesc, Pso):

    def __init__(self):
        Glm.__init__(self)
    
    def _add_bias(self, X):
        """ Add a column of ones to a matrix and return"""
        return np.c_[np.ones(X.shape[0]), X]

    def _ols(self, X, y):
        """ 
        Implement ordinary least square using linear algebra

        Arguments:
        X -- data of size (no_examples, no_features)
        y -- response vector of size (1, no_examples)

        Returns:
        bhat -- model coefficients

        Approach:
        We add the bias term to the X matrix before optimisation
        This is not needed for the other optimisation methods so we do this inside the _ols function
        Note that the shape of X is (no_examples, no_features) where as for gradient descent
        it is currently (no_features, no_examples)
        """
        # add bias 
        X = self._add_bias(X)

        # optimise coefficients
        xTx = np.dot(X.T, X)
        inverse_xTx = np.linalg.inv(xTx)
        xTy = np.dot(X.T, y)
        bhat = np.dot(inverse_xTx, xTy)

        # pull out weights and bias
        b = bhat[0]
        w = bhat[1:]

        return w, b

    def fit(self, X, y, optimiser = 'OLS', loss = 'MSE', num_iterations = 2000, learning_rate = 0.5, debug = False):
        """
        Fit model coefficients

        Arguments:
        X: 1D or 2D numpy array - X.shape = (no_examples, no_features)
            Note we don't class the intercept as a feature. so once we add on the column of ones
            the dimension will be X.shape = (no_examples, no_features + 1)
        y: 1D numpy array y.shape = (no_examples, 1)
        """
        
        # Don't use rank 1 arrays 
        if len(X.shape) == 1:
            X = X.reshape(-1, 1) # https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape
        y = y.reshape(-1, 1)

        # Dimensions of problem        
        self._no_examples, self._no_features =  X.shape 
        
        # train model
        if optimiser == 'OLS': # closed form solution
            w, b = self._ols(X, y)
        elif optimiser == 'GD': # gradient descent
            w, b, debug_mat = self._gd(X.T, y.T, num_iterations = num_iterations, 
                                    learning_rate = learning_rate, loss = loss, debug = debug)
        elif optimiser == 'PSO': # particle swarm
            w, b = self._pso(X, y, no_particles = 300, inertia = 0.9, nostalgia = 1, 
                                  envy = 1, upper = 4, lower = -4)

        assert(w.shape == (self._no_features, 1))
        
        # set attributes
        self.intercept = b
        self.coef = w

        if debug:
            print("iteration: cost: w: b:")
            print("----------------------")
            print(debug_mat)

    def coefficients(self):
        return {'intercept': self.intercept, 'coefficients': self.coef}

    def predict(self, X_new):
        if len(X_new.shape) == 1:
            X_new = X_new.reshape(-1, 1) 
        return self.intercept + np.dot(X_new, self.coef)