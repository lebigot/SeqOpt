# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 13:51:28 2016

@author: A-LAHLOU
"""
import pandas as pd


class cleaner:

    paths = {'data/train_titanic.csv': 'Titanic',
             'data/train_mnist.csv': 'MNIST',
             'data/train_mnist_10k.csv': 'MNIST',
             'data/train_mnist_all.csv': 'MNIST'
             }

    def __init__(self, path_to_train_test=None, header=0):
        self.data_set_name = self.__class__.paths[path_to_train_test]
        self.initial_pd = pd.read_csv(path_to_train_test, header=header)

    def clean(self):
        if self.data_set_name == 'Titanic':
            relevant_cols = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age',
                             'SibSp', 'Parch', 'Fare', 'Embarked']
            to_dummify = ['Sex', 'Embarked']
            contains_na = ['Fare', 'Age']
            identifier = 'PassengerId'
            output = 'Survived'
        if self.data_set_name == 'MNIST':
            self.initial_pd = self.initial_pd.reset_index(drop=False)
            relevant_cols = self.initial_pd.columns.values
            identifier = 'index'
            output = 'label'
            to_dummify = []
            contains_na = []

        train_df = self.initial_pd[relevant_cols]
        if to_dummify:
            train_df = pd.get_dummies(train_df, prefix=to_dummify)
        if contains_na:
            for column in contains_na:
                mean_column = train_df[column].dropna().mean()
                train_df.loc[(train_df[column].isnull()), column] = mean_column
        predictors = [column for column in train_df.columns if column not in
                      {identifier, output}]
        input_data = train_df[predictors]

        input_data = (input_data - input_data.mean()) / (input_data.max() -
                                                         input_data.min())
        # if some columns are constant, the previous operation will result
        # in nan values -> replace them by 0
        input_data = input_data.fillna(0)
        return (input_data, train_df[identifier], train_df[output])
