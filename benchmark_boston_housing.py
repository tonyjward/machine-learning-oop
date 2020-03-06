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

    no_examples, no_features = X.shape

    # LOOCV
    # NO_FOLDS = no_examples
    # folds = np.arange(no_examples)

    # 10-fold
    NO_FOLDS = 10
    np.random.seed(2020)
    folds = np.random.randint(low = 0, high = NO_FOLDS , size = no_examples)

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
        pso_model.fit(X_train, Y_train, optimiser = 'PSO', loss = Mae_pso, num_iterations = 500, no_particles = 300)

        # evaluate models
        OLS_MAE[fold] = evaluate(ols_model, X_test, Y_test, Mae)
        PSO_MAE[fold] = evaluate(pso_model, X_test, Y_test, Mae)
        OLS_RMSE[fold] = evaluate(ols_model, X_test, Y_test, Rmse)
        PSO_RMSE[fold] = evaluate(pso_model, X_test, Y_test, Rmse)

    MAE_diff = OLS_MAE - PSO_MAE
    RMSE_diff = OLS_RMSE- PSO_RMSE

    def sample_std(x):
        return np.sqrt(np.sum(((x - np.mean(x))**2))/(len(x)-1))

    t_statistic = 2.262

    print(f"OLS MAE Average {np.mean(OLS_MAE)}")
    print(f"OLD MAE Confidence interval ({np.mean(OLS_MAE) - t_statistic * sample_std(OLS_MAE)/(NO_FOLDS -1)} , {np.mean(OLS_MAE) + t_statistic * sample_std(OLS_MAE)/(NO_FOLDS -1)}")
    print(f"PSO MAE Average {np.mean(PSO_MAE)}")
    print(f"OLD MAE Confidence interval ({np.mean(PSO_MAE) - t_statistic * sample_std(PSO_MAE)/(NO_FOLDS -1)} , {np.mean(PSO_MAE) + t_statistic * sample_std(PSO_MAE)/(NO_FOLDS -1)}")
    print(f"OLS - PSO: MAE Average {np.mean(MAE_diff)}")
    print(f"OLD - PSO: MAE Confidence interval ({np.mean(MAE_diff) - t_statistic * sample_std(MAE_diff)/(NO_FOLDS -1)} , {np.mean(MAE_diff) + t_statistic * sample_std(MAE_diff)/(NO_FOLDS -1)}")
    print(f"OLS - PSO: RMSE Average {np.mean(RMSE_diff)}")
    print(f"OLD - PSO: RMSE Confidence interval ({np.mean(RMSE_diff) - t_statistic * sample_std(RMSE_diff)/(NO_FOLDS -1)} , {np.mean(RMSE_diff) + t_statistic * sample_std(RMSE_diff)/(NO_FOLDS -1)}")

    print(f"OLS RMSE Average {np.mean(OLS_RMSE)}")
    print(f"PSO RMSE Average {np.mean(PSO_RMSE)}")            


        
        
        





    
    
   
