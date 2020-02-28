from math import floor
import numpy as np

class GradDesc:
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
