import numpy as np

class Glm:
    def __init__(self):
        self.coef = None
        self.intercept = None
        self._no_features = None

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
