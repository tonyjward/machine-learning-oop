from twlearn import LinearRegression
from twlearn.metrics import Mae_pso, Rmse, Mae
import numpy as np
from sklearn.datasets import load_boston
import sklearn.model_selection

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

if __name__ == '__main__':
    
    boston = load_boston()

    X = np.array(boston.data)
    Y = np.array(boston.target)

    NO_FOLDS = 50
    folds = np.random.randint(low = 0, high = NO_FOLDS , size = len(Y))



    OLS_MAE = np.zeros(NO_FOLDS)
    OLS_RMSE = np.zeros(NO_FOLDS)
    PSO_MAE = np.zeros(NO_FOLDS)
    PSO_RMSE = np.zeros(NO_FOLDS)

    for fold in range(NO_FOLDS):
        print(fold)

        X_train = X[folds != fold,:]
        X_test  = X[folds == fold,:]
        Y_train = Y[folds != fold]
        Y_test  = Y[folds == fold]

        # instantiate model objects
        ols_model = LinearRegression()
        pso_model = LinearRegression()

        # train models
        ols_model.fit(X_train, Y_train, optimiser = 'OLS')
        pso_model.fit(X_train, Y_train, optimiser = 'PSO', loss = Mae_pso, num_iterations = 500)

        # evaluate models
        OLS_MAE[fold] = evaluate(ols_model, X_test, Y_test, Mae)
        PSO_MAE[fold] = evaluate(pso_model, X_test, Y_test, Mae)
        OLS_RMSE[fold] = evaluate(ols_model, X_test, Y_test, Rmse)
        PSO_RMSE[fold] = evaluate(pso_model, X_test, Y_test, Rmse)

    print(f"OLS MAE {np.mean(OLS_MAE)}")
    print(f"OLD MAE Confidence intervale ({np.mean(OLS_MAE) - 1.96 * np.std(OLS_MAE)} , {np.mean(OLS_MAE) + 1.96 * np.std(OLS_MAE)}")
    print(f"PSO MAE {np.mean(PSO_MAE)}")
    print(f"OLD MAE Confidence intervale ({np.mean(PSO_MAE) - 1.96 * np.std(PSO_MAE)} , {np.mean(PSO_MAE) + 1.96 * np.std(PSO_MAE)}")
    print(f"OLS RMSE {np.mean(OLS_RMSE)}")
    print(f"PSO RMSE {np.mean(PSO_RMSE)}")

                


        
        
        





    
    
   
