from twlearn import LinearRegression, generate_dataset
from twlearn.metrics import Cautious
import numpy as np

if __name__ == '__main__':
    
    TRIALS = 10
    
    pso_wins_Cautious = 0
 

    for i in range(TRIALS):
        print(i)
        
        X_train, y_train, X_test, y_test, coef = generate_dataset(1000,1000,10, noise = 15, tail_strength = 0.7)
        # OLS
        model_OLS = LinearRegression()
        model_OLS.fit(X_train, y_train)
        model_OLS.coefficients()
        predictions_OLS = model_OLS.predict(X_test)
        print(f"predictions_OLS {predictions_OLS.shape}")
        print(f"y_test {y_test.shape}")
        Cautious_OLS = Cautious(predictions_OLS, y_test)
        print(f"Cautious_OLS {Cautious_OLS.shape}")
        print(f"Cautious Error from OLS model: {Cautious_OLS}")

        # PSO - Cautious
        model_PSO = LinearRegression()
        model_PSO.fit(X_train, y_train, optimiser = 'PSO', loss = Cautious)
        model_PSO.coefficients()
        predictions_PSO = model_PSO.predict(X_test)
        Cautious_PSO = Cautious(predictions_PSO, y_test)
        print(f"Cautious Error from PSO model: {Cautious_PSO}")

        if Cautious_PSO < Cautious_OLS:
            pso_wins_Cautious += 1

    print(f"PSO wins on Cautious: {1.0 * pso_wins_Cautious/ TRIALS}")
