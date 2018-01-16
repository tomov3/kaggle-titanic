import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from load_data import load_test_data, load_train_data
from data_cleaning import clean_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.22, random_state=0)

# Radial Support Vector Machines(rbf-SVM)
model=svm.SVC(kernel='rbf',C=1,gamma=0.1)
model.fit(X_train,y_train)
prediction1=model.predict(X_test)
print('Accuracy for rbf SVM is ',accuracy_score(prediction1,y_test))

# Linear Support Vector Machine(linear-SVM)
model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)
model.fit(X_train,y_train)
prediction2=model.predict(X_test)
print('Accuracy for linear SVM is',accuracy_score(prediction2,y_test))

# Logistic Regression
model=LogisticRegression()
model.fit(X_train,y_train)
prediction3=model.predict(X_test)
print('The accuracy of the Logistic Regression is',accuracy_score(prediction3,y_test))

# Decision Tree
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
prediction4=model.predict(X_test)
print('The accuracy of the Decision Tree is',accuracy_score(prediction4,y_test))

# Perceptron
model=Perceptron()
model.fit(X_train,y_train)
prediction5=model.predict(X_test)
print('The accuracy of the Perceptron is',accuracy_score(prediction5,y_test))

# kNN
model=KNeighborsClassifier(n_neighbors=8) 
model.fit(X_train,y_train)
prediction6=model.predict(X_test)
print('The accuracy of the KNN is',accuracy_score(prediction6,y_test))

# Gaussian Naive Bayes
model=GaussianNB()
model.fit(X_train,y_train)
prediction7=model.predict(X_test)
print('The accuracy of the NaiveBayes is',accuracy_score(prediction7,y_test))

# Random Forest
model=RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)
prediction8=model.predict(X_test)
print('The accuracy of the Random Forests is',accuracy_score(prediction8,y_test))

# Gradient Boosting Classifier
model=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)
model.fit(X_train,y_train)
prediction9=model.predict(X_test)
print('The accuracy of the Gradient Boosting Classifier is',accuracy_score(prediction9,y_test))

# Gradient Boosting Classifier
model=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)
model.fit(X_train,y_train)
prediction10=model.predict(X_test)
print('The accuracy of the Adaboost is',accuracy_score(prediction10,y_test))

# XGBoost
model=xgb.XGBClassifier(n_estimators=900,learning_rate=0.1)
model.fit(X_train,y_train)
prediction10=model.predict(X_test)
print('The accuracy of the XGBoost is',accuracy_score(prediction10,y_test))

# Cross Validation
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
xyz=[]
accuracy=[]
std=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest', 'Gradient Boosting', 'Adaboost', 'XGBoost']
models=[svm.SVC(kernel='linear'),
        svm.SVC(kernel='rbf'),
        LogisticRegression(),
        KNeighborsClassifier(n_neighbors=9),
        DecisionTreeClassifier(),
        GaussianNB(),
        RandomForestClassifier(n_estimators=100),
        GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1),
        AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1),
        xgb.XGBClassifier(n_estimators=900,learning_rate=0.1)]
for i in models:
    model = i
    cv_result = cross_val_score(model, predictors, target, cv = kfold ,scoring = "accuracy")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
print(new_models_dataframe2)

# Hyper Parameter Tuning
from sklearn.model_selection import GridSearchCV
n_estimators=list(range(100,1100,100))
learn_rate=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
hyper={'n_estimators':n_estimators,'learning_rate':learn_rate}
gd=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=hyper,verbose=True)
gd.fit(predictors, target)

# Output Submission file
PassengerId = test['PassengerId']
test_data = test.drop(['PassengerId'], axis=1)

#model = AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)
#model.fit(predictors, target)
#y_pred = model.predict(test_data)
y_pred = gd.predict(test_data)

FILENAME = "../result/adaboost.csv"
sub = pd.DataFrame({'PassengerId' : PassengerId, 'Survived' : y_pred})
sub.to_csv(FILENAME, index=False)
