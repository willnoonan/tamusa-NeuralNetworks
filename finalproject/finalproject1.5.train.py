import torch
from torch import nn
import numpy as np
# from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

from neuralnet import NeuralNet, train, test


train_data = pd.read_csv('data/train.csv')
# data.info()

# Must remove any columns that aren't needed
train_data = train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Parch'], axis=1)
# train_data.head()

# must check for null values, this shows any for each column
train_data.isnull().sum()

# the null check output says there are 177 null values in the age column, and 2 in the Embarked column
# Embarked means which port the ship embarked from, there are only 3 possibilities, {S, C, Q}
# We will just drop the rows where there are null values in the Embarked column:
train_data.dropna(subset=['Embarked'], inplace=True)

# but for the age column, we will fill the null values using the mean of the age column
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)

# now there's no more null values anywhere
train_data.isnull().sum()

# We must convert any text strings to binary values
# we must convert the Sex column to binary: 'male' to 1, 'female' to 0
# create a new column (outside the data DF)
sex_col = train_data['Sex'] == 'male'
sex_col = sex_col.astype('int32')
# let's drop the Sex column in the data DF, we're going to replace it with sex_col
train_data = train_data.drop(['Sex'], axis=1)
# now we replace the Sex column
train_data['Sex'] = sex_col

# next we need to handle the Embarked column, convert it to binary values, we'll use pandas get_dummies function
train_data = pd.get_dummies(train_data, columns=['Embarked'])
# this function expands the Embarked column into three columns of binaries from the unique Embarked values (S, C, Q)

# Now it's time to split the data
X_train = train_data.drop('Survived', axis=1).to_numpy()
y_train = train_data['Survived'].to_numpy()

# Feature scaling used to normalize the range of independent variables or features of data
# feature scaling is applied is that gradient descent converges much faster with feature scaling than without it
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

# split the data
from sklearn.model_selection import train_test_split

# tf.random.set_seed(42)  # will need to import tensorflow stuff

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)



model = NeuralNet()
print(model)

import torch.optim as optim

# train the model
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 10
BATCH_SIZE = 10

train_hist = []
test_hist = []
for epoch in range(n_epochs):
    # Train
    model = train(X_train, y_train, model, loss_fn, optimizer, BATCH_SIZE)

    # Test
    train_loss, train_accuracy = test(X_train, y_train, model, loss_fn)
    test_loss, test_accuracy = test(X_test, y_test, model, loss_fn)

    train_hist.append([train_loss, train_accuracy])
    test_hist.append([test_loss, test_accuracy])

    print(f"Epoch {epoch}: Accuracy {train_accuracy:.2%}, Loss {train_loss:.4}")


# Convert list to numpy arrays
train_hist = np.asarray(train_hist)
test_hist = np.asarray(test_hist)

# Plot loss and accuracy vs epoch
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(train_hist[:, 0], ls='-', marker='.')
plt.plot(test_hist[:, 0], ls='-', marker='.')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(('train', 'test'))

plt.subplot(1, 2, 2)
plt.plot(train_hist[:, 1], ls='-', marker='.')
plt.plot(test_hist[:, 1], ls='-', marker='.')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(('train', 'test'))

plt.tight_layout()
plt.show()
