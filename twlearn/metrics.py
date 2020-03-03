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

def Mae_pso(predictions, actual):
    """
    Calculate Mean Absolute Error

    Arguments:
        predictions: predictions numpy array of size (no_examples, no_particles)
        actuals: 1D numpy array of size (no_examples, 1)

    Returns:
        mae: mae for each particle - numpy array of size (1, no_particles)
    """
    assert(predictions.shape[0] == actual.shape[0])
    absolute_errors = np.abs(predictions - actual)
    return np.mean(absolute_errors, axis = 0, keepdims = True)
