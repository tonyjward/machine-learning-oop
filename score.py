from twlearn import LinearRegression, generate_dataset
from twlearn.metrics import Rmse, Mae

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = generate_dataset(3000,1000,1, noise = 4)
    # OLS
    model_OLS = LinearRegression()
    model_OLS.fit(X_train, y_train)
    model_OLS.coefficients()
    predictions_OLS = model_OLS.predict(X_test)
    RMSE = Rmse(predictions_OLS, y_test)
    MAE = Mae(predictions_OLS, y_test)
    print(f"RMSE from OLS model: {RMSE}")
    print(f"MAE from OLS model: {MAE}")

    # PSO - mae
    model_PSO = LinearRegression()
    model_PSO.fit(X_train, y_train, optimiser = 'PSO')
    model_PSO.coefficients()
    predictions_PSO = model_PSO.predict(X_test)
    RMSE = Rmse(predictions_PSO, y_test)
    MAE = Mae(predictions_PSO, y_test)
    print(f"RMSE from PSO model: {RMSE}")
    print(f"MAE from PSO model: {MAE}")