import numpy as np

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
        return bhat

    def propagate(self, w, b, X, y):
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

        # FORWARD PROPOGATION (FROM X TO COST)
        # Cost function is (1/m)sum((y - a)^2)

        A = b + np.dot(w.T, X)
        cost = (1 / no_examples) * np.sum(np.square(A - y))

        # BACKWARD PROPOGATION (To find gradients)
        dw = (1 / no_examples) * np.dot(X, (A - y).T)
        db = (1 / no_examples) * np.sum(A - y)

        # CHECKS
        assert(dw.shape == w.shape)
    
        grads = {'dw':dw, 'db':db}

        return grads, cost

    def optimise(self, w, b, X, y, num_iterations, learning_rate, print_cost = False):
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
            1) Calculate the cost and the gradient for the current paramters (using propagate)
            2) Update the paramters using gradient descent rule for w and b
        """

        costs = []

        for i in range(num_iterations):
            # cost and gradient calculation
            grads, cost = self.propagate(w, b, X, y)

            # store costs
            if i % 100 == 0:
                costs.append(cost)

            # gradient descent update
            w = w - learning_rate * grads['dw']
            b = b - learning_rate * grads['db']
         
        return w, b, costs   

    def _gd(self, X, y, num_iterations, learning_rate, print_cost):
        """ Build the linear regression model by calling the helper functions

        Arguments:
        X -- training data, numpy array of size (no_features + 1, no_examples)
        Y -- response vector of size (1, no_examples)
        num_iterations -- number of gradient descent iterations
        learning_rate -- learning rate for gradient descent update rule
        print_cost -- logical, should we print the cost function every 100 iterations

        Returns
        return bhat 
        """
  
        # initialise weights
        w, b = self.initialise_with_zeros(self._no_features)

        # optimise weights
        w, b, costs = self.optimise(w, b, X, y, num_iterations, learning_rate, print_cost)

        bhat = np.append(b, w)
        bhat = bhat.reshape(-1, 1)
 
        return bhat

    def initialise_with_zeros(self, dim):
        """
        This function creates a vector of zeros of shape(dim, 1) for w and initialises b to 0

        Argument:
        dim -- size of the w vector we want

        Returns
        w -- initialised vector of shape (dim, 1)
        b -- initialised scalar
        """
        w = np.ones((dim,1))
        b = 0
        return w, b

    def fit(self, X, y, optimiser = 'OLS'):
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
            bhat = self._ols(X, y)
        
        elif optimiser == 'GD': # gradient descent
            bhat = self._gd(X.T, y.T, num_iterations = 2000, learning_rate = 0.5, print_cost = False)
            
        assert(bhat.shape == (self._no_features + 1, 1))
        
        # set attributes
        self.intercept = bhat[0]
        self.coef = bhat[1:]

        assert(self.coef.shape == ((self._no_features, 1)))

    def coefficients(self):
        return {'intercept': self.intercept, 'coefficients': self.coef}

    def predict(self, X_new):
        if len(X_new.shape) == 1:
            X_new = X_new.reshape(-1, 1) 
        return self.intercept + np.dot(X_new, self.coef)