import numpy as np

def Rmse(predictions, actual):
    """
   Calculate Root mean square error

    Arguments:
        predictions: predictions numpy array of size (no_examples, no_solutions)
        actuals: 1D numpy array of size (no_examples, 1)

    Returns:
        rmse: rmse for each solution - numpy array of size (1, no_solutions)
    
    Approach:
    predictions can be a 1d array which corresponds to one set of model predictions OR
    if can be a matrix of predictions, where each column represents a set of predictions
    for a specific model (which is usefule for particle swarm optimisation)
    """
    assert(predictions.shape[0] == actual.shape[0])
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)
    if len(actual.shape) == 1:
        actual = actual.reshape(-1, 1)
   
    no_examples, no_solutions = predictions.shape

    squared_error = np.square(predictions - actual)
    assert(squared_error.shape == predictions.shape)

    sum_of_squares = np.sum(squared_error, axis = 0, keepdims = True)
    assert(sum_of_squares.shape == (1, no_solutions))
    
    rmse = np.sqrt(sum_of_squares)
    assert(rmse.shape == (1, no_solutions))

    return rmse

def Mae(predictions, actual):
    """
    Calculate Mean Absolute Error

    Arguments:
        predictions: predictions numpy array of size (no_examples, no_solutions)
        actuals: 1D numpy array of size (no_examples, 1)

    Returns:
        mae: mae for each solution - numpy array of size (1, no_solutions)
    
    Approach:
    predictions can be a 1d array which corresponds to one set of model predictions OR
    if can be a matrix of predictions, where each column represents a set of predictions
    for a specific model (which is usefule for particle swarm optimisation)
    """
    assert(predictions.shape[0] == actual.shape[0])
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)
    if len(actual.shape) == 1:
        actual = actual.reshape(-1, 1)

    absolute_errors = np.abs(predictions - actual)
    assert(absolute_errors.shape == predictions.shape)

    return np.mean(absolute_errors, axis = 0, keepdims = True)

def fourth_quadrant(predictions, actual, multiplier = 200):
    """
    Calculate Mean Absolute Error

    Arguments:
        predictions: predictions numpy array of size (no_examples, no_particles)
        actuals: 1D numpy array of size (no_examples, 1)
        multiplier: int - how much weight to give underpredictions for positive actual values

    Returns:
        mae: mae for each particle - numpy array of size (1, no_particles)
    """
    assert(predictions.shape[0] == actual.shape[0])

    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)
    if len(actual.shape) == 1:
        actual = actual.reshape(-1, 1)
     
    errors = actual - predictions
    assert(errors.shape == predictions.shape)
 
    negative_error_index = errors < 0
    positive_actual_index = actual > np.mean(actual)

    extra_weight_index = np.logical_and(negative_error_index, positive_actual_index)

    # cautious adjustment
    adjustment = np.ones(shape = errors.shape)
    adjustment[extra_weight_index] = multiplier

    # squared error with adjustment
    adjusted_squared_error = np.multiply(np.square(errors), adjustment)

    return np.mean(adjusted_squared_error, axis = 0, keepdims = True)


