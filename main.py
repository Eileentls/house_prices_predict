import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from data_preparation import generate
from sklearn import ensemble
from sklearn.metrics import r2_score, mean_squared_error


# print R2 and RMSE scores
def __get_score__(predictions, labels):
    print('R2:{}'.format(r2_score(predictions, labels)))
    print('RMSE:{}'.format(np.sqrt(mean_squared_error(predictions, labels))))

# show scores for train and validation sets
def __train_test__(estimator, X_trn, X_tst, y_trn, y_tst):
    print(estimator)
    print('train:')
    predictions_trn = estimator.predict(X_trn)
    __get_score__(predictions_trn, y_trn)
    print('test:')
    predictions_tst = estimator.predict(X_tst)
    __get_score__(predictions_tst, y_tst)

# read train and test files
train_path = './house-prices-advanced-regression-techniques/train.csv'
test_path = './house-prices-advanced-regression-techniques/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# extract label and features from train data
y = train['SalePrice']
X = train.drop(['SalePrice'], axis=1)

(X, X_test) = generate(X, test)
X = X.drop(['Id'], axis=1)
X_test = X_test.drop(['Id'], axis=1)

# split train set and valid set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

GBoost = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt', loss='huber', min_samples_leaf=15, min_samples_split=10).fit(X_train, y_train)

__train_test__(GBoost, X_train, X_valid, y_train, y_valid)
score = cross_val_score(GBoost, X, y, cv=5)
print('Accuracy: %0.2f (+- %0.2f)'% (score.mean(), score.std()*2))

final_labels = GBoost.predict(X_test)

# saving to csv
pd.DataFrame({'Id' : test.Id, 'SalePrice' : final_labels}).to_csv('prediction.csv', index=False)




# print(X_train.head())
# X_train.to_csv('x_train.csv')
# print(X_train.describe())
# print(X_valid.describe())
