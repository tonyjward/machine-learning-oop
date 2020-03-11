from twlearn import LinearRegression
from twlearn.metrics import Rmse, Mae, fourth_quadrant
from twlearn.statistics import five_by_two_cv, p_value
from twlearn.utility import evaluate, cross_validate

import numpy as np
from sklearn.datasets import load_boston
import sklearn.model_selection

NO_FOLDS = 2
NO_REPEATS = 5
CUSTOM_LOSS = fourth_quadrant

if __name__ == '__main__':
       

    CUSTOM_LOSS = fourth_quadrant

    # Train Models
    print("Training OLS Model")
    ols_results_4Q = cross_validate(no_repeats = NO_REPEATS, no_folds = NO_FOLDS, loss = CUSTOM_LOSS, 
                                kwargs = {"optimiser":"OLS"})
    print("Training PSO Model")
    pso_results_4Q = cross_validate(no_repeats = NO_REPEATS, no_folds = NO_FOLDS, loss = CUSTOM_LOSS,
                             kwargs = {"optimiser":'PSO', "loss":CUSTOM_LOSS, "num_iterations":5000, "no_particles":500})

    # Cross Validation Results
    print("Comparing Model Error")
    for repeat in range(NO_REPEATS):
        for fold in range(NO_FOLDS):
            print(f"Error for repeat {repeat} fold {fold}: OLS: {np.round(ols_results_4Q[repeat][fold],2)} PSO: {np.round(pso_results_4Q[repeat][fold],2)} difference : {np.round(ols_results_4Q[repeat][fold] - pso_results_4Q[repeat][fold],2)}")

    # Significance Testing
    t_statistic, average_differences = five_by_two_cv(ols_results_4Q, pso_results_4Q)

    for repeat in average_differences:
        print(f"Average difference for repeat {repeat}: {np.round(average_differences[repeat],4)}")

    p = p_value(t_statistic = t_statistic,
                degrees_freedom = 5,
                sided = 2)

    print(f"\n The t statistic is {np.round(t_statistic,2)} which has a p value of {p}")


   
    
    
           


        
        
        





    
    
   
