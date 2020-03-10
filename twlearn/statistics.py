import numpy as np
from scipy import stats

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

def p_value(t_statistic, degrees_freedom, sided):
    """
    Arguments:
        t_statistic -- float -- students test statistic
        degrees_freedom -- int - degrees of freedom for test statistic
        sided -- int (1 or 2) - how many sides should the test have

    Returns:
        p_value -- float -- p - value associated with the test

    """
    assert sided in (1,2), 'sided must be 1 or 2'

    p_one_sided = 1 - stats.t.cdf(t_statistic, df=degrees_freedom)
    p_two_sided = 2 * p_one_sided

    if sided == 1:
        return p_one_sided
    elif sided == 2:
        return p_two_sided



