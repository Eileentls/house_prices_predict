import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# observing sale price histogram
# sns.distplot(y)
# plt.show()

# concatenate train and test and drop features that do not correlate to SalePrice
# data_features = pd.concat([X, test], keys=['train', 'test'])
# print('data_feature.train[0]:', '\n', data_features.loc['train'].loc[0])
def __fill_missing_values__(data_features):
    data_features.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'LowQualFinSF',
               'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',
               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'],
                   axis=1, inplace=True)
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
    data_features['Electrical'] = data_features['Electrical'].fillna(data_features['Electrical'].mode().iloc[0])

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
    data_features['TotalSF'] = data_features['TotalBsmtSF'] + data_features['1stFlrSF'] + data_features['2ndFlrSF']
    data_features.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)
    return data_features


# extract categorical features and numerical features
# categorical_features = data_features.select_dtypes(include=['object']).columns
# print('categorical_features:', '\n', categorical_features)
# print('categorical_features data:', '\n', data_features[categorical_features].head())
# numerical_features = data_features.select_dtypes(exclude=['object']).columns
# print('numerical_features:', '\n', numerical_features)
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

# NAs = __count_missing_values__(data_features.loc['train'], data_features.loc['test'])
# print(NAs)
# __plot_features__(numerical_features, data_features.loc['train'], y)

# standardized. Numerical feature
def __num_features_standardized(train, test, numerical_features_standardized):
    std_scaler = StandardScaler()
    train_standardized = pd.DataFrame(std_scaler.fit_transform(train[numerical_features_standardized]))
    test_standardized= pd.DataFrame(std_scaler.transform(test[numerical_features_standardized]))
    return (train_standardized, test_standardized)

# __num_features_standardized(data_features.loc['train'], data_features.loc['test'], numerical_features_standardized)

# categorical features. label encoding, one-hot encoding.
def __cate_features_transform__(train, test, categorical_features, label_features):
    label_encoder = LabelEncoder()
    for col in label_features:
        train[col] = label_encoder.fit_transform(train[col])
        test[col] = label_encoder.transform(test[col])
    OH_features = [val for val in categorical_features if val not in label_features]
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_train = pd.DataFrame(OH_encoder.fit_transform(train[OH_features]))
    OH_test = pd.DataFrame(OH_encoder.transform(test[OH_features]))
    train_transform = pd.concat([train[label_features], OH_train], axis=1)
    test_transform = pd.concat([test[label_features], OH_test], axis=1)
    return (train_transform, test_transform)

def generate(train, test):
    train_with_missing = __fill_missing_values__(train)
    test_with_missing = __fill_missing_values__(test)

    numerical_features_standardized = ['LotFrontage', 'LotArea', 'GrLivArea', 'TotalSF']
    (train_standardized, test_standardized) = __num_features_standardized(train_with_missing, test_with_missing, numerical_features_standardized)

    categorical_features = test.select_dtypes(include=['object']).columns
    # categorical_features = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
    #    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
    #    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
    #    'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
    #    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    #    'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
    #    'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
    #    'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
    #    'SaleType', 'SaleCondition']

    label_features = ['LotShape', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageFinish', 'GarageQual',
                      'SaleCondition']
    (train_transform, test_transform) = __cate_features_transform__(train_with_missing, test_with_missing, categorical_features, label_features)
    train_drop = train_with_missing.drop(categorical_features, axis=1).drop(numerical_features_standardized, axis=1)
    test_drop = test_with_missing.drop(categorical_features, axis=1).drop(numerical_features_standardized, axis=1)
    new_train = pd.concat([train_drop, train_standardized, train_transform], axis=1)
    new_test = pd.concat([test_drop, test_standardized, test_transform], axis=1)
    return (new_train, new_test)

