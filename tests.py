from twlearn import MyLinearRegression, generate_dataset

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = generate_dataset(10,10,2)
    model = MyLinearRegression()
    model.fit(X_train, y_train)
    model.coef()
