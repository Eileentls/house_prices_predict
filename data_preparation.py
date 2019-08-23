import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

#read train and test files
train_path = './house-prices-advanced-regression-techniques/train.csv'
test_path = './house-prices-advanced-regression-techniques/test.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
# print(train_data.head())
# print(test_data.head())

#extract label and features from train data
y = train_data.SalePrice
X = train_data.drop(columns=['SalePrice'])

# split train set and valid set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
# print(X_train.describe())
# print(X_valid.describe())

#count missing values in train data and test data
NAs = pd.concat([X.isnull().sum(), test_data.isnull().sum()], axis=1, keys=['train', 'test'])
print(NAs[NAs.sum(axis=1) > 0])
print(NAs[NAs.sum(axis=1) > 0].index)
