import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, ParameterGrid

from load_data import load_test_data, load_train_data
from data_cleaning import clean_data

def acc(y, yPred):
    return np.sum(yPred==y)/len(y)

# laod data
train_data = clean_data(load_train_data())
train_data.drop(['PassengerId'], axis=1, inplace=True)
test_data = clean_data(load_test_data())

x_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']

# training
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
all_params = {'C': [10**i for i in range(-3, 3)],
                  'fit_intercept': [True, False],
                  'penalty': ['l2', 'l1'],
                  'random_state': [0]}
max_score = -1
max_params = None

for params in tqdm(list(ParameterGrid(all_params))):
    list_acc_score = []
    for train_idx, valid_idx in cv.split(x_train, y_train):
        trn_x = x_train.iloc[train_idx, :]
        val_x = x_train.iloc[valid_idx, :]

        trn_y = y_train[train_idx]
        val_y = y_train[valid_idx]

        clf = LogisticRegression(**params)
        clf.fit(trn_x, trn_y)
        pred = clf.predict(val_x)

        sc_acc = acc(val_y, pred)

        list_acc_score.append(sc_acc)

    sc_acc = np.mean(list_acc_score)
    if max_score < sc_acc:
        max_score = sc_acc
        max_params = params

print("max_score: {}".format(max_score))
print("max_params: {}".format(max_params))

clf = LogisticRegression(**max_params)
clf.fit(x_train, y_train)
print("score: {}".format(clf.score(x_train, y_train)))

PassengerId = test_data[test_data.columns[0]]
x_test = test_data[test_data.columns[1:]]

predictions = clf.predict(x_test)
sub = pd.DataFrame({ 'PassengerId': PassengerId,
                                'Survived': predictions })
sub.to_csv("../result/lr.csv", index=False)
