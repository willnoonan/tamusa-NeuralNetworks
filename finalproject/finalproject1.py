import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

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


