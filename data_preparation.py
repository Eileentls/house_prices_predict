import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

# read train and test files
train_path = './house-prices-advanced-regression-techniques/train.csv'
test_path = './house-prices-advanced-regression-techniques/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
# print(train_data.head())
# print(test_data.head())

# extract label and features from train data
y = train['SalePrice']
X = train.drop(['SalePrice'], axis=1)
print(X.describe())
print(X.shape)

# observing sale price histogram
# sns.distplot(y)
# plt.show()

# concatenate train and test
data_features = pd.concat([X, test])
print(data_features.shape)
print(data_features.head())

# extract categorical features and numerical features
categorical_features = data_features.select_dtypes(include=['object']).columns
print('categorical_features:', '\n', categorical_features)
print('categorical_features data:', '\n', data_features[categorical_features].head())
numerical_features = data_features.select_dtypes(exclude=['object']).columns
print('numerical_features:', '\n', numerical_features)
print('numerical_features data:', '\n', data_features[numerical_features].head())

# plot the features
def _plot_features(features, data_features, data_label):
    figures_per_time = 4
    count = 0
    for var in features:
        plt.figure(count//figures_per_time, figsize=(25, 5))
        plt.subplot(1, figures_per_time, np.mod(count, 4)+1)
        plt.scatter(data_features[var], data_label)
        plt.title('f model T= {}'.format(var))
        count += 1
    plt.show()


# count missing values in train and test
def _count_missing_values(train, test):
    NAs = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['train', 'test'])
    return NAs[NAs.sum(axis=1) > 0]

NAs = _count_missing_values(X, test)
# print(NAs)
num_features_with_missing = [index for index in NAs.index
                             if index in numerical_features]
# print(num_features_with_missing)
cate_features_with_missing = [index for index in NAs.index
                              if index in categorical_features]
# print(cate_features_with_missing)

# deal with missing values
for col in num_features_with_missing:
    data_features[col] = data_features[col].fillna(0)
    X[col] = X[col].fillna(X[col].mean())
    test[col] = test[col].fillna(test[col].mean())

# print(data_features[num_features_with_missing].isnull().sum())

_plot_features(numerical_features, X, y)


# split train set and valid set
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
# print(X_train.describe())
# print(X_valid.describe())
