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

# extract label and features from train data
y = train['SalePrice']
X = train.drop(['SalePrice'], axis=1)

# observing sale price histogram
# sns.distplot(y)
# plt.show()

# concatenate train and test and drop features that do not correlate to SalePrice
data_features = pd.concat([X, test], keys=['train', 'test'])
# print('data_feature.train[0]:', '\n', data_features.loc['train'].loc[0])
data_features.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'LowQualFinSF',
               'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',
               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'],
                   axis=1, inplace=True)


# extract categorical features and numerical features
categorical_features = data_features.select_dtypes(include=['object']).columns
# print('categorical_features:', '\n', categorical_features)
# print('categorical_features data:', '\n', data_features[categorical_features].head())
numerical_features = data_features.select_dtypes(exclude=['object']).columns
# print('numerical_features:', '\n', numerical_features)
# print('numerical_features data:', '\n', data_features[numerical_features].head())

# plot the numerical features
def _plot_features(features, data_features, data_label):
    figures_per_time = 4
    count = 0
    for var in features:
        plt.figure(count//figures_per_time, figsize=(25, 5))
        plt.subplot(1, figures_per_time, np.mod(count, 4)+1)
        plt.scatter(data_features[var], data_label)
        plt.title('f model T= {}'.format(var))
        count += 1
    # plt.show()


# count missing values in train and test
def _count_missing_values(train, test):
    NAs = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['train', 'test'])
    return NAs[NAs.sum(axis=1) > 0]

NAs = _count_missing_values(data_features.loc['train'], data_features.loc['test'])
print(NAs)
# num_features_with_missing = [index for index in NAs.index
#                              if index in numerical_features]
# print('numerical_features_with_missing:', '\n', num_features_with_missing)
# cate_features_with_missing = [index for index in NAs.index
#                               if index in categorical_features]
# print('categorical_features_with_missing:', '\n', cate_features_with_missing)

# fill NAs in features
# MSZoning. NAs in test. fill with mode
data_features['MSZoning'] = data_features['MSZoning'].fillna(data_features['MSZoning'].mode())

# LotFrontage. Numerical feature. I suppose NAs mean 0
data_features['LotFrontage'] = data_features['LotFrontage'].fillna(0)

# Alley. NA means no alley access
data_features['Alley'] = data_features['Alley'].fillna('NOACCESS')

# Exterior1st, Exterior2nd. NA in test. fill with mode
for col in ['Exterior1st', 'Exterior2nd']:
    data_features[col] = data_features[col].fillna(data_features[col].mode())


# print(data_features[num_features_with_missing].isnull().sum())

_plot_features(numerical_features, data_features.loc['train'], y)


# split train set and valid set
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
# print(X_train.describe())
# print(X_valid.describe())
