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
    
        print('***** PREPROCESSING END *****')


if __name__ == "__main__":
    t = Titanic()
    t.hook()