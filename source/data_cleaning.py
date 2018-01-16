import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def clean_data(data):
    #data=pd.read_csv('../input/train.csv')

    data['Initial']=0
    for i in data:
        data['Initial']=data.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations

    data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                            ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Mrs'],
                            inplace=True)

    ## Assigning the NaN Values with the Ceil values of the mean ages
    data.loc[(data.Age.isnull())&(data.Initial=='Mr'),'Age']=33
    data.loc[(data.Age.isnull())&(data.Initial=='Mrs'),'Age']=36
    data.loc[(data.Age.isnull())&(data.Initial=='Master'),'Age']=5
    data.loc[(data.Age.isnull())&(data.Initial=='Miss'),'Age']=22
    data.loc[(data.Age.isnull())&(data.Initial=='Other'),'Age']=46

    data['Embarked'].fillna('S',inplace=True)

    data['Age_band']=0
    data.loc[data['Age']<=16,'Age_band']=0
    data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1
    data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2
    data.loc[(data['Age']>48)&(data['Age']<=64),'Age_band']=3
    data.loc[data['Age']>64,'Age_band']=4

    data['Family_Size']=1
    data['Family_Size']=data['Parch']+data['SibSp']+1#family size
    data['Alone']=0
    data.loc[data.Family_Size==1,'Alone']=1#Alone

    data['Fare_Range']=pd.qcut(data['Fare'],4)

    data['Fare_cat']=0
    data.loc[data['Fare']<=7.91,'Fare_cat']=0
    data.loc[(data['Fare']>7.91)&(data['Fare']<=14.454),'Fare_cat']=1
    data.loc[(data['Fare']>14.454)&(data['Fare']<=31),'Fare_cat']=2
    data.loc[(data['Fare']>31)&(data['Fare']<=513),'Fare_cat']=3

    data['Sex'].replace(['male','female'],[0,1],inplace=True)
    data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
    data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)

    data['Cabin_Bool'] = data['Cabin'].notnull().astype('int')

    data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range'],axis=1,inplace=True)

    return data
