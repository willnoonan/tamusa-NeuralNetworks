"""
Author: William Noonan
Linear Regression (LR) using Gradient Descent (GD)
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed



def lin_reg_gd(X, Y, learning_rate=1, epochs=3000):
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


def mse_of_lin_reg_gd(X, Y, **kwargs):
    m, b = lin_reg_gd(X, Y, **kwargs)
    y_pred = X * m + b
    return mean_squared_error(Y, y_pred)


# Load the diabetes data
data = datasets.load_diabetes(as_frame=True)
Y = data.target
df = data.data

# res = {col: mse_of_lin_reg_gd(df[col], Y) for col in df.columns}
res = [(col, mse_of_lin_reg_gd(df[col], Y)) for col in df.columns]
# results = dict(sorted(res.items(), key=lambda item: item[1]))
sorted_by_second = sorted(res, key=lambda tup: tup[1])
