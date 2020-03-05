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

def Cautious(predictions, actual):
    """
    Calculate Mean Absolute Error

    Arguments:
        predictions: predictions numpy array of size (no_examples, no_particles)
        actuals: 1D numpy array of size (no_examples, 1)

    Returns:
        mae: mae for each particle - numpy array of size (1, no_particles)
    """
    assert(predictions.shape[0] == actual.shape[0])

    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)
    if len(actual.shape) == 1:
        actual = actual.reshape(-1, 1)
     
    errors = predictions - actual
    assert(errors.shape == predictions.shape)
 
    negative_error_index = errors < 0
    positive_actual_index = actual > 0

    extra_weight_index = np.logical_and(negative_error_index, positive_actual_index)

    # cautious adjustment
    adjustment = np.ones(shape = errors.shape)
    adjustment[extra_weight_index] = 5

    # squared error with adjustment
    cautious_squared_error = np.multiply(np.square(errors), adjustment)

    return np.mean(cautious_squared_error, axis = 0, keepdims = True)



