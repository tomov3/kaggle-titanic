import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from load_data import load_test_data, load_train_data
from data_cleaning import clean_data

# laod data
train_data = clean_data(load_train_data())
train_data.drop(['PassengerId'], axis=1, inplace=True)
test_data = clean_data(load_test_data())

# split training data into training/testing sets
train,test=train_test_split(train_data,test_size=0.3,random_state=0,stratify=train_data['Survived'])
train_X=train[train.columns[1:]]
train_y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_y=test[test.columns[:1]]
X=train_data[train_data.columns[1:]]
y=train_data['Survived']

# training
all_params = {'C': [10**i for i in range(-3, 3)],
                  'fit_intercept': [True, False],
                  'penalty': ['l2', 'l1'],
                  'random_state': [0]}
gd = GridSearchCV(estimator=LogisticRegression(), param_grid=all_params, verbose=True)
gd.fit(X, y)

PassengerId = test_data[test_data.columns[0]]
testX = test_data[test_data.columns[1:]]

predictions = gd.predict(testX)
sub = pd.DataFrame({ 'PassengerId': PassengerId,
                                'Survived': predictions })
sub.to_csv("../result/lr.csv", index=False)
