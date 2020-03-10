from twlearn import LinearRegression
from twlearn.metrics import Rmse, Mae, Cautious
from twlearn.statistics import five_by_two_cv
import numpy as np
from sklearn.datasets import load_boston
import sklearn.model_selection

NO_FOLDS = 2
NO_REPEATS = 5

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
    Approach:
        We want the random numbers for repeat 1 to be different for repeat 2, etc
        however we also want the random numbers for repeat 1, to be the same
        each time we run the cross_validation to ensure we can compare algorithm
        performance. We therefore set the seed to be the repeat number.
    """  
    boston = load_boston()

    X = np.array(boston.data)
    Y = np.array(boston.target)

    no_examples, no_features = X.shape

    cv_results = {}

    for repeat in range(NO_REPEATS):
        print(f"repeat: {repeat}")
        
        np.random.seed(repeat)
        folds = np.random.randint(low = 0, high = NO_FOLDS , size = no_examples)
        
        cv_results[repeat] = {}
  
        for fold in range(NO_FOLDS):
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

if __name__ == '__main__':
    
    CUSTOM_LOSS = Cautious

    print("Training OLS Model")
    ols_results = cross_validate(no_repeats = NO_REPEATS, no_folds = NO_FOLDS, loss = CUSTOM_LOSS, 
                                 kwargs = {"optimiser":"OLS"})
    
    print("OLS Results")
    print(ols_results)

    print("Training PSO Model")
    pso_results = cross_validate(no_repeats = NO_REPEATS, no_folds = NO_FOLDS, loss = CUSTOM_LOSS,
                                 kwargs = {"optimiser":'PSO', "loss":CUSTOM_LOSS, "num_iterations":5000, "no_particles":500})

    print("PSO Results")
    print(pso_results)

    print(f" t_statistic {five_by_two_cv(ols_results, pso_results)}")

   
    
    
           


        
        
        





    
    
   
