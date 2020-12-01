import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.utils import shuffle



class Titanic:

    def __init__(self):
        self.path = os.path.abspath('../data')
        self.train = self.read_train()
        self.test = self.read_test()

        self.train_label = object
        self.train_data = object
        self.test_data = object


    def hook(self):
        
        # print('***** SEX PIE CHART CHECK *****')
        # self.pie_chart('Sex')

        # print('***** PCLASS PIE CHART CHECK *****')
        # self.pie_chart('Pclass')

        # print('***** EMBARKED PIE CHART CHECK *****')
        # self.pie_chart('Embarked')

        # print('***** SIBSP BAR CHART CHECK *****')
        # self.bar_chart('SibSp')

        # print('***** PARCH BAR CHART CHECK *****')
        # self.bar_chart('Parch')

        '''
        성별이 여성일 수록(영화 타이타닉에서 나온 것 처럼 여성과 아이부터 먼저 살렸기 때문이 아닐까 싶고),
        Pclass가 높을 수록(맨 위의 사진을 보면 타이타닉 호는 배의 후미부터 잠기기 시작되었다는 것을 알 수 있는데,
         티켓의 등급이 높아질 수록 숙소가 배의 앞쪽과 위쪽으로 가는 경향이 있어 그 영향이 아닐까 싶고),
        Cherbourg 선착장에서 배를 탔다면,
        형제, 자매, 배우자, 부모, 자녀와 함께 배에 탔다면,
        생존 확률이 더 높았다는 것을 볼 수 있다.
        '''
        self.preprocessing()

        self.modeling()
    
    def read_train(self):
        print('***** READ TRAIN DATA START *****')
        train = pd.read_csv(self.path + '\\train.csv')
        print(train.info())
        print('***** READ TRAIN DATA END *****')
        return train

    def read_test(self):
        print('***** READ TEST DATA START *****')
        test = pd.read_csv(self.path + '\\test.csv')
        print(test.info())
        print('***** READ TEST DATA END *****')
        return test

    def pie_chart(self, feature):
        train = self.train
        feature_ratio = train[feature].value_counts(sort=False)
        feature_size = feature_ratio.size
        feature_index = feature_ratio.index
        survived = train[train['Survived'] == 1][feature].value_counts()
        dead = train[train['Survived'] == 0][feature].value_counts()

        plt.plot(aspect='auto')
        plt.pie(feature_ratio, labels=feature_index, autopct='%1.1f%%')
        plt.title(feature + '\'s ratio in total')
        plt.show()

        for i, index in enumerate(feature_index):
            plt.subplot(1, feature_size + 1, i + 1, aspect='equal')
            plt.pie([survived[index], dead[index]], labels=['Survived', 'Dead'], autopct='%1.1f%%')
            plt.title(str(index) + '\'s ratio')
        
        plt.show()
    
    def bar_chart(self, feature):
        train = self.train
        survived = train[train['Survived'] == 1][feature].value_counts()
        dead = train[train['Survived'] == 0][feature].value_counts()
        df = pd.DataFrame([survived, dead])
        df.index = ['Survived', 'Dead']
        df.plot(kind='bar', stacked=True, figsize=(10,5))

        plt.show()

    def preprocessing(self):
        print('***** PREPROCESSING START *****')

        train = self.train
        test = self.test
        train_and_test = [train, test]

        for dataset in train_and_test:
            dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.')
        
        print(train.head())

        t = pd.crosstab(train['Title'], train['Sex'])
        print(t)

        for dataset in train_and_test:
            dataset['Title'] = dataset['Title'].replace(['Capt',
                                                        'Col',
                                                        'Countess',
                                                        'Don',
                                                        'Dona',
                                                        'Dr',
                                                        'Jonkheer',
                                                        'Lady',
                                                        'Major',
                                                        'Rev',
                                                        'Sir'],
                                                        'Other')

            dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
            dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
            dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

        a = train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
        print(a)

        for dataset in train_and_test:
            dataset['Title'] = dataset['Title'].astype(str)

        for dataset in train_and_test:
            dataset['Sex'] = dataset['Sex'].astype(str)
    
        b = train.Embarked.value_counts(dropna=False)
        print(b)
        
        for dataset in train_and_test:
            dataset['Embarked'] = dataset['Embarked'].fillna('S')
            dataset['Embarked'] = dataset['Embarked'].astype(str)

        for dataset in train_and_test:
            dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
            dataset['Age'] = dataset['Age'].astype(int)
            train['AgeBand'] = pd.cut(train['Age'], 5)
        print(train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())

        for dataset in train_and_test:
            dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
            dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
            dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
            dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
            dataset.loc[dataset['Age'] >= 64, 'Age'] = 4
            dataset['Age'] = dataset['Age'].map( {0:'Child', 1:'Young', 2:'Middle', 3:'Prime', 4:'Lod', }).astype(str)

        print(train[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).mean())
        print("")
        print(test[test['Fare'].isnull()]['Pclass'])

        for dataset in train_and_test:
            dataset['Fare'] = dataset['Fare'].fillna(13.675)

        for dataset in train_and_test:
            dataset.loc[dataset['Fare'] <= 7.854, 'Fare'] = 0
            dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1
            dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare'] = 2
            dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare'] = 3
            dataset.loc[dataset['Fare'] >= 39.688, 'Fare'] = 4
            dataset['Fare'] = dataset['Fare'].astype(int)

        for dataset in train_and_test:
            dataset['Family'] = dataset['Parch'] + dataset['SibSp']
            dataset['Family'] = dataset['Family'].astype(int)

        features_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
        train = train.drop(features_drop, axis=1)
        test = test.drop(features_drop, axis=1)
        train = train.drop(['PassengerId', 'AgeBand'], axis=1)

        train = pd.get_dummies(train)
        test = pd.get_dummies(test)

        train_label = train['Survived']
        train_data = train.drop('Survived', axis=1)
        test_data = test.drop('PassengerId', axis=1).copy()

        # One-hot-encoding for categorical variables
        train = pd.get_dummies(train)
        test = pd.get_dummies(test)

        train_label = train['Survived']
        train_data = train.drop('Survived', axis=1)
        test_data = test.drop("PassengerId", axis=1).copy()

        self.train_label = train_label
        self.train_data = train_data
        self.test_data = test_data

        print(train.head())
        print(train.info())

        print(test.head())
        print(test.info())

        print('*'*30)
        print('***** PREPROCESSING END *****')

    def modeling(self):
        print('***** MODELING START *****')

        train_label = self.train_label
        train_data = self.train_data 
        test_data = self.test_data

        train_data, train_label = shuffle(train_data, train_label, random_state = 5)
        print(train_data)
        print(train_label)

        print('***** MODELING END *****')

if __name__ == "__main__":
    t = Titanic()
    t.hook()