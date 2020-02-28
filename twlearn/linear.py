import numpy as np
from math import floor

class LinearRegression:
    def __init__(self):
        self.coef = None
        self.intercept = None
        self._no_features = None
   
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

    def _propagate(self, w, b, X, y, loss):
        """
        Implement the cost function and its gradient

        Arguments:
        w -- weights, a numpy array of size (number of features, 1)
        b -- bias, a scalar
        X -- data of size (number of features, number of examples)
        y -- response vector of size (1, number of examples)

        Return:
        cost -- negative log likelihood cost for linear regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b
        """
        # TODO: check dimensions
        no_features, no_examples = X.shape

        A = b + np.dot(w.T, X)

        if loss == 'MSE':
            # FORWARD PROPOGATION (FROM X TO COST)
            # Cost function is (1/m)sum((y - a)^2)
            cost = (1 / no_examples) * np.sum(np.square(A - y))

            # BACKWARD PROPOGATION (To find gradients)
            dw = (1 / no_examples) * np.dot(X, (A - y).T)
            db = (1 / no_examples) * np.sum(A - y)
        elif loss == 'MAE':
            # FORWARD PROPOGATION (FROM X TO COST)
            # Cost function is (1/m)sum(|y - a|)
            cost = (1 / no_examples) * np.sum(np.absolute(y - A))

            # BACKWARD PROPOGATION (To find gradients)
            part1 = np.sum(X[:, ((y - A) < 0).flatten()], axis = 1)
            part2 = np.sum(X[:, ((y - A) > 0).flatten()], axis = 1)
                      
            dw = (1 / no_examples) * (part1 - part2)
            dw = dw.reshape(-1, 1)
            db = (1 / no_examples) * (np.sum((y - A) > 0) - np.sum((y - A) < 0)) 
        
        
        assert(dw.shape == w.shape)     
    
        grads = {'dw':dw, 'db':db}

        return grads, cost

    def _optimise(self, w, b, X, y, num_iterations, learning_rate, loss, debug):
        """
        This function optimises w and b by running a gradient descent algorithm

        Arguments:
        w -- weights, a numpy array of size (no_features, 1)
        b -- bias, a scalar
        X -- training data, numpy array of size (no_features, no_examples)
        y -- response vector of size (1, no_examples)
        num_iterations -- number of iterations of the optimisation loop
        learning_rate -- learing rate of the gradient descent update rule

        Returns:
        w -- weights, a numpy array of size (no_features, 1)
        b -- bias, a scalar
        costs -- a vector of costs for every 100th iteration of gradient descent 

        Approach:
            1) Calculate the cost and the gradient for the current paramters (using _propagate)
            2) Update the paramters using gradient descent rule for w and b
        """
        # debug matrix with columns (iteration, costs, w, b)

        SAVE_EVERY = 100
        debug_rows = floor(num_iterations / SAVE_EVERY)
        debug_cols = 3 + w.shape[0]
        debug_mat = np.zeros((debug_rows, debug_cols))

        for i in range(num_iterations):
            # cost and gradient calculation
            grads, cost = self._propagate(w, b, X, y, loss)

            # store costs
            if i % SAVE_EVERY == 0:
                row = int(i / SAVE_EVERY)
                debug_mat[row, 0] = i
                debug_mat[row, 1] = cost
                debug_mat[row, 2:(2+w.shape[0])] = np.round(w.flatten(), 3)
                debug_mat[row, 2 + w.shape[0]] = np.round(b, 3)

            # gradient descent update
            w = w - learning_rate * grads['dw']
            b = b - learning_rate * grads['db']
         
        return w, b, debug_mat   

    def _gd(self, X, y, num_iterations, learning_rate, loss, debug):
        """ Build the linear regression model by calling the helper functions

        Arguments:
        X -- training data, numpy array of size (no_features + 1, no_examples)
        Y -- response vector of size (1, no_examples)
        num_iterations -- number of gradient descent iterations
        learning_rate -- learning rate for gradient descent update rule
        print_cost -- logical, should we print the cost function every 100 iterations

        Returns
        w -- weights
        b -- bias 
        """

        # initialise weights
        w, b = self._initialise_with_zeros(self._no_features)

        # optimise weights
        w, b, debug_mat = self._optimise(w, b, X, y, num_iterations, learning_rate, loss, debug)

        return w, b, debug_mat

    def _initialise_with_zeros(self, dim):
        """
        This function creates a vector of zeros of shape(dim, 1) for w and initialises b to 0

        Argument:
        dim -- size of the w vector we want

        Returns
        w -- initialised vector of shape (dim, 1)
        b -- initialised scalar
        """
        w = np.ones((dim,1)) * 100
        b = 100
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
        no_examples, self._no_features =  X.shape 
        
        # train model
        if optimiser == 'OLS': # closed form solution
            w, b = self._ols(X, y)
        
        elif optimiser == 'GD': # gradient descent
            w, b, debug_mat = self._gd(X.T, y.T, num_iterations = num_iterations, learning_rate = learning_rate, loss = loss, debug = debug)

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