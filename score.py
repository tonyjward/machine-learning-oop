from twlearn import LinearRegression, generate_dataset
from twlearn.metrics import Rmse, Mae
import numpy as np

if __name__ == '__main__':
    
    TRIALS = 100
    
    pso_wins_RMSE = 0
    pso_wins_MAE = 0
    for i in range(TRIALS):
        print(i)
        
        X_train, y_train, X_test, y_test, coef = generate_dataset(1000,1000,10, noise = 15, tail_strength = 0.7)
    
        # OLS
        model_OLS = LinearRegression()
        model_OLS.fit(X_train, y_train)
        model_OLS.coefficients()
        predictions_OLS = model_OLS.predict(X_test)
        RMSE_OLS = Rmse(predictions_OLS, y_test)
        MAE_OLS = Mae(predictions_OLS, y_test)
        # print(f"RMSE from OLS model: {RMSE_OLS}")
        print(f"MAE from OLS model: {MAE_OLS}")

        # PSO - mae
        model_PSO = LinearRegression()
        model_PSO.fit(X_train, y_train, optimiser = 'PSO')
        model_PSO.coefficients()
        predictions_PSO = model_PSO.predict(X_test)
        RMSE_PSO = Rmse(predictions_PSO, y_test)
        MAE_PSO = Mae(predictions_PSO, y_test)
        # print(f"RMSE from PSO model: {RMSE_PSO}")
        print(f"MAE from PSO model: {MAE_PSO}")

        if RMSE_PSO < RMSE_OLS:
            pso_wins_RMSE += 1
        
        if MAE_PSO < MAE_OLS:
            pso_wins_MAE += 1
    
    print(f"PSO wins on RMSE: {1.0 * pso_wins_RMSE/ TRIALS}")
    print(f"PSO wins on MAE: {1.0 * pso_wins_MAE/ TRIALS}")