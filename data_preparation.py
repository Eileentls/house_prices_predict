import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

#read train and test files
train_path = './house-prices-advanced-regression-techniques/train.csv'
test_path = './house-prices-advanced-regression-techniques/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
# print(train_data.head())
# print(test_data.head())

# extract label and features from train data
y = train.SalePrice
X = train.drop(columns=['SalePrice'])
print(y.describe())

# observing sale price histogram
sns.distplot(y)
# plt.show()

# concatenate train and test
data_features = pd.concat((X, test)).reset_index(drop=True)
print(data_features.shape)

# extract categorical features and numerical features
categorical_features = data_features.select_dtypes(include=['object']).columns
print('categorical_features:', '\n', categorical_features)
numerical_features = data_features.select_dtypes(exclude=['object']).columns
print('numerical_features:', '\n', numerical_features)

# plot the numerical features
features = numerical_features
figures_per_time = 4
count = 0
for var in features:
    x = train[var]
    plt.figure(count//figures_per_time, figsize=(25, 5))
    plt.subplot(1, figures_per_time, np.mod(count, 4)+1)
    plt.scatter(x, y)
    plt.title('f model T= {}'.format(var))
    count += 1

# plt.show()


# split train set and valid set
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
# print(X_train.describe())
# print(X_valid.describe())

# count missing values in train data and test data
# NAs = pd.concat([X.isnull().sum(), test.isnull().sum()], axis=1, keys=['train', 'test'])
# print(NAs[NAs.sum(axis=1) > 0])
# print(NAs[NAs.sum(axis=1) > 0].index)
