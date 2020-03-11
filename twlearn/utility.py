from sklearn.datasets import load_boston
import numpy as np

from twlearn import LinearRegression

def evaluate(model, X_test, Y_test, loss):
    """ Take a model and and some test data and produce test metrics
    Arguments:
        model         -- a fitted model
        X_test        -- test data - a numpy array
        loss_function -- a loss function to assess predictions

    Returns:
        loss - calculated loss
    """
    predictions = model.predict(X_test)
    return loss(predictions, Y_test)
    
def cross_validate(no_repeats, no_folds, loss, kwargs):
    """
    Arguments:
        no_repeats - int - number of cross validation repeats
        no_folds - int - number of cross validation folds
        loss - function - loss function to evaluate predictions
        kwargs - dict - settings required for the fit function of OLS and PSO implementations
        
    Returns:
        cv_results: dict - cross validation results
        
    Approach:
        We want the random numbers for repeat 1 to be different for repeat 2, etc
        however we also want the random numbers for repeat 1, to be the same
        each time we run the cross_validation to ensure we can compare algorithm
        performance. We therefore set the seed to be the repeat number.
    """  
    boston = load_boston()

    X = np.array(boston.data)
    Y = np.log(np.array(boston.target))

    no_examples, no_features = X.shape

    cv_results = {}

    for repeat in range(no_repeats):
        print(f"repeat: {repeat}")
        
        np.random.seed(repeat)
        folds = np.random.randint(low = 0, high = no_folds , size = no_examples)
        
        cv_results[repeat] = {}
  
        for fold in range(no_folds):
            print(f"fold: {fold}")

            X_train = X[folds != fold,:]
            X_test  = X[folds == fold,:]
            Y_train = Y[folds != fold]
            Y_test  = Y[folds == fold]           

            # train model
            model = LinearRegression()
            model.fit(X_train, Y_train, **kwargs)

            # evaluate model
            cv_results[repeat][fold] = evaluate(model, X_test, Y_test, loss)

    return cv_results