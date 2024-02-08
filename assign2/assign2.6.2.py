"""
Author: William Noonan
Linear Regression (LR) using Gradient Descent (GD)
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import random


def lin_reg_gd(X, Y, learning_rate=1, epochs=5000):
    """
    Computes slope and y-intercept (m,b) of linear regression fit using gradient descent.
    :param epochs: Number of learning steps
    :param learning_rate: step size
    :param X:
    :param Y:
    :return: slope and y-intercept (m, b)
    """
    m, b = 0, 0  # initial values
    n = float(len(X))  # Number of elements in X
    # Performing Gradient Descent
    for i in range(epochs):
        Y_pred = m * X + b  # The current predicted value of Y
        m = m - (learning_rate * (1 / n) * np.sum(X * (Y_pred - Y)))  # Update m
        b = b - (learning_rate * (1 / n) * np.sum(Y_pred - Y))  # Update b

    return m, b

class Result:
    def __init__(self, m, b, mse, test):
        self.m = m
        self.b = b
        self.mse = mse
        x, y = zip(*test)
        self.testX = np.array(x)
        self.testY = np.array(y)

    def __repr__(self):
        return f'<Result: mse: {self.mse}>'


def mse_of_lin_reg_gd(X, Y, **kwargs):
    # Zip X and Y
    zipXY = list(zip(X, Y))

    # split the data, 70% train, 15% dev, 15% test
    split_1 = int(0.70 * len(zipXY))
    split_2 = int(0.85 * len(zipXY))
    train = zipXY[:split_1]
    dev = zipXY[split_1:split_2]
    test = zipXY[split_2:]

    
    x_train, y_train = zip(*train)
    x_dev, y_dev = zip(*dev)

    # train the model
    m, b = lin_reg_gd(np.array(x_train), np.array(y_train), **kwargs)

    # make predictions on dev set
    y_dev_pred = np.array(x_dev) * m + b
    mse = mean_squared_error(np.array(y_dev), y_dev_pred)

    res = Result(m, b, mse, test)

    return res

from sklearn.utils import shuffle as sk_shuffle
def main():
    # Load the diabetes data
    data = datasets.load_diabetes(as_frame=True)

    df_shuffled = sk_shuffle(data.frame, random_state=1234)
    features = df_shuffled.drop(columns='target')
    Y = df_shuffled.target.values

    with ProcessPoolExecutor() as executor:
        futures = {col: executor.submit(mse_of_lin_reg_gd, features[col], Y) for col in features.columns}
        # Wait for futures to finish
        for _ in as_completed(futures.values()):
            pass
        results = sorted([(key, value.result()) for key, value in futures.items()], key=lambda tup: tup[1].mse)

    # Extract Result with smallest mse
    selected, res = results[0]
    print(f"The variable selected through auto-detection is '{selected}'.")
    # Use the model
    m, b = res.m, res.b
    testX, testY = res.testX, res.testY
    pred_test_Y = testX * m + b
    print(f"MSE on test is {mean_squared_error(testY, pred_test_Y)}")

    # sklearn lin reg
    diabetes_X = testX[:, np.newaxis]  # must reshape for regr.fit

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X, testY)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X)

    """
    Plots
    """
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    # Lin reg from scratch w/GD
    ax1.scatter(testX, testY)
    ax1.plot(testX, pred_test_Y, color='red')  # regression line
    ax1.set_title('LR using GD')

    # Sklearn lin reg model
    ax2.scatter(testX, testY)  # same result if you use X_fi
    ax2.plot(testX, diabetes_y_pred, color="blue")
    ax2.set_title('sklearn LR model')

    plt.show()


# The following prevents RuntimeError for 'spawn' and 'forkserver' start_methods:
if __name__ == '__main__':
    main()