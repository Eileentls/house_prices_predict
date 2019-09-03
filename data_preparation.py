import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

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
print('categorical_features:', '\n', categorical_features)
# print('categorical_features data:', '\n', data_features[categorical_features].head())
numerical_features = data_features.select_dtypes(exclude=['object']).columns
print('numerical_features:', '\n', numerical_features)
# print('numerical_features data:', '\n', data_features[numerical_features].head())

# plot the numerical features
def __plot_features__(features, data_features, data_label):
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
def __count_missing_values__(train, test):
    NAs = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['train', 'test'])
    return NAs[NAs.sum(axis=1) > 0]

NAs = __count_missing_values__(data_features.loc['train'], data_features.loc['test'])
print(NAs)

# fill NAs in features
# MSZoning. NAs in test. fill with mode
data_features['MSZoning'] = data_features['MSZoning'].fillna(data_features['MSZoning'].mode().iloc[0])

# LotFrontage. NAs in all. Numerical feature. I suppose NAs mean 0
data_features['LotFrontage'] = data_features['LotFrontage'].fillna(0)

# Alley. NAs in all. NA means no alley access
data_features['Alley'] = data_features['Alley'].fillna('NOACCESS')

# Exterior1st, Exterior2nd. NA in test. fill with mode
for col in ['Exterior1st', 'Exterior2nd']:
    data_features[col] = data_features[col].fillna(data_features[col].mode().iloc[0])

# MasVnrType. NAs in all. Fill with mode.
data_features['MasVnrType'] = data_features['MasVnrType'].fillna(data_features['MasVnrType'].mode().iloc[0])

# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2. NAs in all. NA means no basement.
for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:
    data_features[col] = data_features[col].fillna('NOBSMT')

# TotalBsmtSF. NAs in test. I suppose NA means 0.
data_features['TotalBsmtSF'] = data_features['TotalBsmtSF'].fillna(0)

# Electrical. NAs in train. Fill with mode.
data_features['Electrical'] = data_features['Electsrical'].fillna(data_features['Electrical'].mode().iloc[0])

# KitchenQual. NA in test. Fill with mode.
data_features['KitchenQual'] = data_features['KitchenQual'].fillna(data_features['KitchenQual'].mode().iloc[0])

# FireplaceQu. NAs in all. NA means no fireplace.
data_features['FireplaceQu'] = data_features['FireplaceQu'].fillna('NOFP')

# GarageType, GarageFinish, GarageQual. NAs in all. NA means no garage.
for col in ['GarageType', 'GarageFinish', 'GarageQual']:
    data_features[col] = data_features[col].fillna('NOGRG')

# GarageCars. NA in test. I suppose NA means 0.
data_features['GarageCars'] = data_features['GarageCars'].fillna(0)

# SaleType. NA in test. Fill with mode.
data_features['SaleType'] = data_features['SaleType'].fillna(data_features['SaleType'].mode().iloc[0])

# Add total square feet, drop basement, 1st, 2nd floor features
# data_features['TotalSF'] = data_features['TotalBsmtSF'] + data_features['1stFlrSF'] + data_features['2ndFlrSF']
# data_features.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)

__plot_features__(numerical_features, data_features.loc['train'], y)

# pipeline standardized. Numerical features.
# one-hot encoding. categorical features.
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
# print(data_features[categorical_features].head())
OH_data_features = pd.DataFrame(OH_encoder.fit_transform(data_features[categorical_features]))
# data_features_tmp = data_features.drop(categorical_features, axis=1)
# data_features = pd.concat([OH_data_features, data_features_tmp], axis=1)
# print(data_features.loc['train'].columns)
print(OH_data_features.columns)

def generate(train, test, label):
    return
