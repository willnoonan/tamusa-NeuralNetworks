"""
Author: William Noonan
Linear Regression (LR) using Gradient Descent (GD)
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error


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


# Load the diabetes data
data = datasets.load_diabetes(as_frame=True)
df = data.frame

# Take a look at correlation matrix, sort by most to least correlated in target col
target_corr = df.corr().target.sort_values(ascending=False)
most_corr_feature = ""
for index in target_corr.index:
    if index != 'target':
        most_corr_feature = index
        break

if not most_corr_feature:
    raise Exception("Could not find most correlated feature.")
else:
    print(f"The most correlated feature is '{most_corr_feature}'")


# Grab data of most correlated feature
X_fi = df.bmi
Y = df.target

"""
Linear regression from scratch w/gradient descent
"""
m, b = lin_reg_gd(X_fi, Y)
y_pred = X_fi * m + b  # prediction of y using the computed m, b

print(f"MSE: {mean_squared_error(Y, y_pred)}")

"""
From sklearn's Linear Regression Example
https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
"""
diabetes_X = X_fi[:, np.newaxis]  # must reshape for regr.fit

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X, Y)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X)

"""
Plots
"""
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

# Lin reg from scratch w/GD
ax1.scatter(X_fi, Y)
ax1.plot(X_fi, y_pred, color='red')  # regression line
ax1.set_title('LR using GD')

# Sklearn lin reg model
ax2.scatter(diabetes_X, Y)  # same result if you use X_fi
ax2.plot(diabetes_X, diabetes_y_pred, color="blue")
ax2.set_title('sklearn LR model')

plt.show()
