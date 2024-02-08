from stdimports import *
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

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

X = torch.tensor(X_train, dtype=torch.float32)
y = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)


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

model = NeuralNet()
print(model)

import torch.optim as optim

# train the model
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 10
batch_size = 10

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i + batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i + batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')

# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy * 100}")