import numpy as np

def Rmse(predicted, actual):
    """
    Calculate root mean square error

    Arguments:
    predicted: 1D numpy array
    actual: 1D numpy array
    """
    sum_of_squares = np.sum(np.square(predicted - actual))
    return np.sqrt(sum_of_squares)

def Mae(predicted, actual):
    """
    Calculate Mean Absolute Error

    Arguments:
    predicted: 1D numpy array
    actual: 1D numpy array
    """
    absolute_errors = np.abs(predicted - actual)
    return np.mean(absolute_errors)
