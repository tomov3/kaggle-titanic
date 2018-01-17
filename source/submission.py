import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from load_data import load_test_data, load_train_data
from data_cleaning import clean_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb

train = clean_data(load_train_data())
test = clean_data(load_test_data())

predictors = train.drop(['PassengerId', 'Survived'], axis=1)
target = train['Survived']

xgboost_param = {
        'learning_rate':[0.1],
        'n_estimators':[1000],
        'max_depth':[3,5],
        'min_child_weight':[1,2,3],
        'max_delta_step':[5],
        'gamma':[0,3,10],
        'subsample':[0.8],
        'colsample_bytree':[0.8],
        'objective':['binary:logistic'],
        'nthread':[4],
        'scale_pos_weight':[1],
        'seed':[0]}

rforest_param = {'max_depth': [2,3,4,5,6,7,8,9,10,11,12],
    'min_samples_split' :[2,3,4,5,6],
    'n_estimators' : [i for i in range(10, 100, 10)],
    'min_samples_leaf': [1,2,3,4,5],
    'max_features': (2,3,4)}

#gd=GridSearchCV(estimator=xgb.XGBClassifier(),param_grid=xgboost_param,verbose=True, n_jobs=-1)
gd=GridSearchCV(estimator=RandomForestClassifier(),param_grid=rforest_param,verbose=True, n_jobs=-1)
gd.fit(predictors, target)

PassengerId = test['PassengerId']
test_data = test.drop(['PassengerId'], axis=1)
y_pred = gd.predict(test_data)

FILENAME = '../result/submit.csv'
sub = pd.DataFrame({'PassengerId' : PassengerId, 'Survived' : y_pred})
sub.to_csv(FILENAME, index=False)
