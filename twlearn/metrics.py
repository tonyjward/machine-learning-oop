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

def Cautious(predictions, actual, multiplier = 100):
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
     
    errors = predictions - actual
    assert(errors.shape == predictions.shape)
 
    negative_error_index = errors < 0
    positive_actual_index = actual > 0

    extra_weight_index = np.logical_and(negative_error_index, positive_actual_index)

    # cautious adjustment
    adjustment = np.ones(shape = errors.shape)
    adjustment[extra_weight_index] = multiplier

    # squared error with adjustment
    cautious_squared_error = np.multiply(np.square(errors), adjustment)

    return np.mean(cautious_squared_error, axis = 0, keepdims = True)

def five_by_two_cv(errorA, errorB, no_repeats = 5, no_folds = 2):
        """ compute the 5x2cv paired t test
        Arguments:
             
            errorA: dictionary - results of 5 repeats of 2 fold cross validation for model A   
            errorB: dictionary - results of 5 repeats of 2 fold cross validation for model B
            no_repeats: int    - number of repeats of cross validaiton
            no_folds: int      - number of folds in each repeat of cross validation
                   
        Returns:
            the 5x2cv paired t test as described in https://sci2s.ugr.es/keel/pdf/algorithm/articulo/dietterich1998.pdf 
        Approach:
            Arguments errorA and error B contain the results of 5 repeats of two fold cross validation
            for algorithm A and B respectively.
        """

        # check dimensions of errorA and errorB are in keeping with 5 repeats of 2 fold cv
        assert(errorA.keys() == errorB.keys())
        assert(len(errorA.keys()) == 5)
        for repeat in range(5):
            assert len(errorA[repeat].keys()) == 2
            assert len(errorB[repeat].keys()) == 2

        differences = {}

        # differences between models
        for repeat in range(no_repeats):
            differences[repeat] = {}
            for fold in range(no_folds):
                differences[repeat][fold] = errorA[repeat][fold] - errorB[repeat][fold]

        # average differences
        average_differences = {}
        for repeat in range(no_repeats):
            average_differences[repeat] = (differences[repeat][0] + differences[repeat][1]) / 2

        # variances
        variances = {}
        for repeat in range(no_repeats):
            variances[repeat] = (differences[repeat][0] - average_differences[repeat]) ** 2 + (differences[repeat][1] - average_differences[repeat]) ** 2
       
        sum_of_squares = 0
        for repeat in range(no_repeats):
            sum_of_squares += variances[repeat]

        t_statistic = differences[0][0] / np.sqrt((1 / no_repeats) * sum_of_squares)

        return t_statistic, average_differences

