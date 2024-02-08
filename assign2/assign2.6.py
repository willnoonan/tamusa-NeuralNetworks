"""
Author: William Noonan
Linear Regression (LR) using Gradient Descent (GD)
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


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
    """
    Class used to bundle up results
    """

    def __init__(self, m, b, mse, test):
        self.m = m
        self.b = b
        self.mse = mse
        x, y = zip(*test)
        self.testX = np.array(x)
        self.testY = np.array(y)

    def __repr__(self):
        # nice for debugging
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

    # unpack train and dev sets
    x_train, y_train = zip(*train)
    x_dev, y_dev = zip(*dev)

    # train the model using train set
    m, b = lin_reg_gd(np.array(x_train), np.array(y_train), **kwargs)

    # make predictions using dev set
    y_dev_pred = np.array(x_dev) * m + b

    # get mean squared error for dev set
    mse = mean_squared_error(np.array(y_dev), y_dev_pred)

    # bundle up results in Result instance
    res = Result(m, b, mse, test)
    return res


from sklearn.utils import shuffle as sk_shuffle


def main():
    """
    Model training and automatic feature selection
    """
    # Load the diabetes data
    data = datasets.load_diabetes(as_frame=True)

    # Shuffle the DataFrame
    df_shuffled = sk_shuffle(data.frame, random_state=1234)
    features = df_shuffled.drop(columns='target')  # extract the features
    Y = df_shuffled.target.values  # extract the target values

    # multiprocessing to speed things up
    with ProcessPoolExecutor() as executor:
        futures = {col: executor.submit(mse_of_lin_reg_gd, features[col], Y) for col in features.columns}

        # wait for futures to finish
        for _ in as_completed(futures.values()):
            pass

        # extract results into list of tuples (<feature>, <Result instance>), then sort by mse
        results = sorted([(key, value.result()) for key, value in futures.items()], key=lambda tup: tup[1].mse)

    """
    Test the model
    """
    # unpack the feature, Result instance with smallest mse
    selected, res = results[0]
    print(f"The feature selected through auto-detection is '{selected}'.")

    # extract the associated trained model's slope and y-intercept, and test set
    m, b = res.m, res.b
    testX, testY = res.testX, res.testY

    # predict Y on test set using trained model's slope, y-intercept
    pred_test_Y = testX * m + b

    # compute MSE of test set Y and predicted test Y
    print(f"MSE for test set is {mean_squared_error(testY, pred_test_Y)}")

    """
    Plot
    """
    plt.scatter(testX, testY, label='test set')
    plt.plot(testX, pred_test_Y, label='LR-GD prediction', color='red')
    plt.xlabel(selected)
    plt.ylabel('target')
    plt.legend()
    plt.savefig('assign2plot.png')
    plt.show()


# The following prevents RuntimeError for 'spawn' and 'forkserver' start_methods:
if __name__ == '__main__':
    main()
