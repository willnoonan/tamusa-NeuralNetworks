import torch
from torch import nn
import numpy as np
# from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

train_csv_path = 'data/train.csv'

data = pd.read_csv(train_csv_path)
# data.info()

# Must remove any columns that aren't needed
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Parch'], axis=1)
data.head()

# must check for null values, this shows any for each column
data.isnull().sum()

# the null check output says there are 177 null values in the age column, and 2 in the Embarked column
# Embarked means which port the ship embarked from, there are only 3 possibilities, {S, C, Q}
# We will just drop the rows where there are null values in the Embarked column:
data.dropna(subset=['Embarked'], inplace=True)

# but for the age column, we will fill the null values using the mean of the age column
data['Age'].fillna(data['Age'].mean(), inplace=True)

# now there's no more null values anywhere
data.isnull().sum()

# We must convert any text strings to binary values
# we must convert the Sex column to binary: 'male' to 1, 'female' to 0
# create a new column (outside the data DF)
sex_col = data['Sex'] == 'male'
sex_col = sex_col.astype('int32')

# let's drop the Sex column in the data DF, we're going to replace it with sex_col
data = data.drop(['Sex'], axis=1)

# now we replace the Sex column
data['Sex'] = sex_col

# next we need to handle the Embarked column, convert it to binary values, we'll use pandas get_dummies function
data = pd.get_dummies(data, columns=['Embarked'])
# this function expanded the Embarked column into three columns of binaries from the unique Embarked values (S, C, Q)

# Now it's time to split the data
X = data.drop('Survived', axis=1).to_numpy()
y = data['Survived'].to_numpy()

# Feature scaling used to normalize the range of independent variables or features of data
# feature scaling is applied is that gradient descent converges much faster with feature scaling than without it
sc = StandardScaler()
X = sc.fit_transform(X)

# split the data
from sklearn.model_selection import train_test_split

# tf.random.set_seed(42)  # will need to import tensorflow stuff

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using %s device" % DEVICE)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 12)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(12, 8)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(8, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x


def train(X, y, model, loss_fn, optimizer, verbose=True):
    model.train()
    for i in range(0, len(X), BATCH_SIZE):
        Xbatch = X[i:i + BATCH_SIZE].to(DEVICE)
        ybatch = y[i:i + BATCH_SIZE].to(DEVICE)
        y_pred = model(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model  # latest loss is returned



def test(X, y, model, loss_fn):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
    accuracy = (y_pred.round() == y).float().mean()
    return loss, accuracy



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
    model = train(X_train, y_train, model, loss_fn, optimizer)

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
