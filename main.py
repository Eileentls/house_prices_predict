import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_preparation import generate

# read train and test files
train_path = './house-prices-advanced-regression-techniques/train.csv'
test_path = './house-prices-advanced-regression-techniques/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# extract label and features from train data
y = train['SalePrice']
X = train.drop(['SalePrice'], axis=1)

(X_train, X_test) = generate(X, test)

# split train set and valid set
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y, test_size=0.2, random_state=0)

print(X_train.head())
# X_train.to_csv('x_train.csv')
# print(X_train.describe())
# print(X_valid.describe())
