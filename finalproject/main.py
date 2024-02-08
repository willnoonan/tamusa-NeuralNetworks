import torch
from torch import nn
import numpy as np
# from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim


from neuralnet import NeuralNet, train, test


train_data = pd.read_csv('data/train.csv')

# Remove any columns that aren't needed
train_data = train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin',], axis=1)

# check for null values, this shows any for each column
train_data.isnull().sum()

# there are 177 null values in the age column, and 2 in the Embarked column
# drop the rows where there are null Embarked values
train_data.dropna(subset=['Embarked'], inplace=True)

# for age column, fill the null values using the mean age
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)

# now there's no more null values anywhere
train_data.isnull().sum()

# We must convert any text strings to binary values
# we must convert the Sex column to binary: 'male' to 1, 'female' to 0
# create a new column (outside the data DF)
sex_col = train_data['Sex'] == 'male'
sex_col = sex_col.astype('int32')
# drop the Sex column in the dataframe, replace it with sex_col
train_data = train_data.drop(['Sex'], axis=1)
# now we replace the Sex column
train_data['Sex'] = sex_col

# expand the Embarked column into three columns of binaries based on its unique values (S, C, Q)
train_data = pd.get_dummies(train_data, columns=['Embarked'])

# Now it's time to split the data
X = train_data.drop('Survived', axis=1).to_numpy()
y = train_data['Survived'].to_numpy()

# Feature scaling used to normalize the range of independent variables or features of data
sc = StandardScaler()
X = sc.fit_transform(X)

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# convert numpy arrays to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


# create model instance
model = NeuralNet()
print(model)

# train the model
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 10
BATCH_SIZE = 10

train_hist = []
test_hist = []
for epoch in range(n_epochs):
    # training
    model = train(X_train, y_train, model, loss_fn, optimizer, BATCH_SIZE)

    # testing
    train_loss, train_accuracy = test(X_train, y_train, model, loss_fn)
    test_loss, test_accuracy = test(X_test, y_test, model, loss_fn)

    train_hist.append([train_loss, train_accuracy])
    test_hist.append([test_loss, test_accuracy])

    print(f"Epoch {epoch}: Train Accuracy {train_accuracy:.2%}, Train Loss {train_loss:.4}"
          f"Test Accuracy {test_accuracy:.2%}, Test Loss {test_loss:.4}")


# Convert history lists to numpy arrays
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
# plt.savefig('traincsv-plot.png')


"""
Now to process the data in test.csv for submission
Repeat most the preprocessing done for train.csv
Create a new model called test_model
"""
test_data = pd.read_csv('data/test.csv')

# hold on to PassengerId data in new dataframe
passenger = test_data[['PassengerId']]

# drop columns
test_data = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', ], axis=1)

# check for nulls
test_data.isnull().sum()

# replace nulls in Age and Fare columns with their means
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

# convert Sex column to binary
sex_col = test_data['Sex'] == 'male'
sex_col = sex_col.astype('int32')
test_data = test_data.drop(['Sex'], axis=1)
# now we replace the Sex column with binary series
test_data['Sex'] = sex_col

# expand Embarked to 3 columns of binary values
test_data = pd.get_dummies(test_data, columns=['Embarked'])

sc = StandardScaler()
X_test = sc.fit_transform(test_data)

# convert to tensor
X_test = torch.tensor(X_test, dtype=torch.float32)

# use entire training dataset for training
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# new model instance
test_model = NeuralNet()

loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(test_model.parameters(), lr=0.001)

n_epochs = 50
BATCH_SIZE = 10

train_hist = []
test_hist = []
for epoch in range(n_epochs):
    # train the model
    test_model = train(X, y, test_model, loss_fn, optimizer, BATCH_SIZE)

    # test the model
    train_loss, train_accuracy = test(X, y, test_model, loss_fn)

    train_hist.append([train_loss, train_accuracy])

    print(f"Epoch {epoch}: Accuracy {train_accuracy:.2%}, Loss {train_loss:.4}")

# Convert list to numpy arrays
train_hist = np.asarray(train_hist)

# Plot loss and accuracy vs epoch
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(train_hist[:, 0], ls='-', marker='.')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(train_hist[:, 1], ls='-', marker='.')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')


plt.tight_layout()
plt.show()
# plt.savefig('testcsv-plot.png')

# Make predictions on the test.csv data
with torch.no_grad():
    y_pred = test_model(X_test)

# insert predictions into the DataFrame with PassengerId data
passenger['Survived'] = y_pred.round().int()

# export it to csv
passenger.to_csv('submission.csv', index=False)
