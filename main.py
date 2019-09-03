import pandas as pd
import numpy as np
from data_preparation import generate

# read train and test files
train_path = './house-prices-advanced-regression-techniques/train.csv'
test_path = './house-prices-advanced-regression-techniques/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

generate(train, test, label)



# split train set and valid set
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
# print(X_train.describe())
# print(X_valid.describe())
