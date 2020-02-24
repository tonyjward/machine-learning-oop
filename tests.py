from twlearn import MyLinearRegression, generate_dataset
from twlearn.metrics import Rmse, Mae

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = generate_dataset(10,10,2)
    model = MyLinearRegression()
    model.fit(X_train, y_train)
    model.coefficients()
    predictions = model.predict(X_test)
    RMSE = Rmse(predictions, y_test)
    MAE = Mae(predictions, y_test)
    print(f"RMSE: {RMSE}")
    print(f"MAE: {MAE}")
