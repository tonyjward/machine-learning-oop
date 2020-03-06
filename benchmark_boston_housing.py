from twlearn import LinearRegression
from twlearn.metrics import Mae_pso, Rmse, Mae
import numpy as np
from sklearn.datasets import load_boston
import sklearn.model_selection

NO_FOLDS = 2
NO_REPEATS = 5

np.random.seed(2020)

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

    OLS_MAE = {}
    PSO_MAE = {}

    for repeat in range(NO_REPEATS):
        print(f"repeat: {repeat}")
        folds = np.random.randint(low = 0, high = NO_FOLDS , size = no_examples)
        OLS_MAE[repeat] = {}
        PSO_MAE[repeat] = {}

        for fold in range(NO_FOLDS):
            print(f"fold: {fold}")

            X_train = X[folds != fold,:]
            X_test  = X[folds == fold,:]
            Y_train = Y[folds != fold]
            Y_test  = Y[folds == fold]

            # instantiate model objects
            ols_model = LinearRegression()
            pso_model = LinearRegression()

            # train models
            ols_model.fit(X_train, Y_train, optimiser = 'OLS')
            pso_model.fit(X_train, Y_train, optimiser = 'PSO', loss = Mae_pso, num_iterations = 2000, no_particles = 500)

            # evaluate models
            OLS_MAE[repeat][fold] = evaluate(ols_model, X_test, Y_test, Mae)
            PSO_MAE[repeat][fold] = evaluate(pso_model, X_test, Y_test, Mae)

    def five_by_two_cv(errorA, errorB):
        """ compute the 5x2cv paired t test
        Arguments:
             
            errorA: dictionary - results of 5 repeats of 2 fold cross validation for model A   
            errorB: dictionary - results of 5 repeats of 2 fold cross validation for model B
                   
        Returns:
            the 5x2cv paired t test as described in https://sci2s.ugr.es/keel/pdf/algorithm/articulo/dietterich1998.pdf 
        Approach:
            Arguments errorA and error B contain the results of 5 repeats of two fold cross validation
            for algorithm A and B respectively.
        """
        differences = {}

        # differences between models
        for repeat in range(NO_REPEATS):
            differences[repeat] = {}
            for fold in range(NO_FOLDS):
                differences[repeat][fold] = errorA[repeat][fold] - errorB[repeat][fold]

        # average differences
        average_differences = {}
        for repeat in range(NO_REPEATS):
            average_differences[repeat] = (differences[repeat][0] + differences[repeat][0]) / 2

        # variances
        variances = {}
        for repeat in range(NO_REPEATS):
            variances[repeat] = (differences[repeat][0] - average_differences[repeat]) ** 2 + (differences[repeat][1] - average_differences[repeat]) ** 2
       
        sum_of_squares = 0
        for repeat in range(NO_REPEATS):
            sum_of_squares += variances[repeat]

        t_statistic = differences[0][0] / np.sqrt((1 / NO_REPEATS) * sum_of_squares)


        return t_statistic, average_differences





    
    
    print(f" t_statistic {five_by_two_cv(OLS_MAE, PSO_MAE)}")
    
    
    
    
    
    
    # MAE_diff = OLS_MAE - PSO_MAE
    # RMSE_diff = OLS_RMSE- PSO_RMSE

    # test_statistic = five_by_two_cv(OLS_MAE, PSO_MAE)
    # print(OLS_MAE)
    # print(f"5by2 statistic: {test_statistic}")
    # print(f"MAE differences: {MAE_diff}")
    # print(f"RMSE differences: {RMSE_diff}")
    # print(f"OLS MAE Average {np.mean(OLS_MAE)}")
    # print(f"OLS MAE Confidence interval ({np.mean(OLS_MAE) - t_statistic * sample_std(OLS_MAE)/(NO_FOLDS -1)} , {np.mean(OLS_MAE) + t_statistic * sample_std(OLS_MAE)/(NO_FOLDS -1)}")
    # print(f"PSO MAE Average {np.mean(PSO_MAE)}")
    # print(f"PSO MAE Confidence interval ({np.mean(PSO_MAE) - t_statistic * sample_std(PSO_MAE)/(NO_FOLDS -1)} , {np.mean(PSO_MAE) + t_statistic * sample_std(PSO_MAE)/(NO_FOLDS -1)}")
    # print(f"OLS - PSO: MAE Average {np.mean(MAE_diff)}")
    # print(f"OLD - PSO: MAE Confidence interval ({np.mean(MAE_diff) - t_statistic * sample_std(MAE_diff)/(NO_FOLDS -1)} , {np.mean(MAE_diff) + t_statistic * sample_std(MAE_diff)/(NO_FOLDS -1)}")
    # print(f"OLS - PSO: RMSE Average {np.mean(RMSE_diff)}")
    # print(f"OLD - PSO: RMSE Confidence interval ({np.mean(RMSE_diff) - t_statistic * sample_std(RMSE_diff)/(NO_FOLDS -1)} , {np.mean(RMSE_diff) + t_statistic * sample_std(RMSE_diff)/(NO_FOLDS -1)}")

    #print(f"OLS RMSE Average {np.mean(OLS_RMSE)}")
    #print(f"PSO RMSE Average {np.mean(PSO_RMSE)}")            


        
        
        





    
    
   
