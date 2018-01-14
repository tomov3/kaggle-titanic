import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import AdaBoostClassifier
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
train_Y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]
X=train_data[train_data.columns[1:]]
Y=train_data['Survived']

# Hyper-Parameter Tuning for AdaBoost
n_estimators=list(range(100,1100,100))
learn_rate=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
hyper={'n_estimators':n_estimators,'learning_rate':learn_rate}
gd=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=hyper,verbose=True)
gd.fit(X,Y)

PassengerId = test_data[test_data.columns[0]]
testX = test_data[test_data.columns[1:]]

predictions = gd.predict(testX)
sub = pd.DataFrame({ 'PassengerId': PassengerId,
                                'Survived': predictions })
sub.to_csv("../result/adaboost.csv", index=False)
