import torch
from torch import nn
import numpy as np
# from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

from neuralnet import NeuralNet, train, test


test_data = pd.read_csv('data/test.csv')
# data.info()

# Test set
test_data = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Parch'], axis=1)

# check for nulls
test_data.isnull().sum()

# there's null values in Age and Fare cols
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

sex_col = test_data['Sex'] == 'male'
sex_col = sex_col.astype('int32')
# let's drop the Sex column in the data DF, we're going to replace it with sex_col
test_data = test_data.drop(['Sex'], axis=1)
# now we replace the Sex column
test_data['Sex'] = sex_col

# next we need to handle the Embarked column, convert it to binary values, we'll use pandas get_dummies function
test_data = pd.get_dummies(test_data, columns=['Embarked'])

sc = StandardScaler()
X_test = sc.fit_transform(test_data)


X_test = torch.tensor(X_test, dtype=torch.float32)


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
    model = train(X_train, y_train, model, loss_fn, optimizer, batch_size=BATCH_SIZE)

    # Test
    train_loss, train_accuracy = test(X_train, y_train, model, loss_fn)
    test_loss, test_accuracy = test(X_test, y_test, model, loss_fn)

    print(f"Epoch {epoch}: Accuracy {train_accuracy:.2%}, Loss {train_loss:.4}")

    train_hist.append([train_loss, train_accuracy])
    test_hist.append([test_loss, test_accuracy])


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
