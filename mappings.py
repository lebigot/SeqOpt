# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 10:39:33 2016

@author: A-LAHLOU
"""
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GB
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss


search_methods = {'GS': GridSearchCV, 'RS': RandomizedSearchCV}
scores = {'AUC': roc_auc_score, 'ACC': accuracy_score, 'LL': log_loss}
algos = {'RF': RF, 'GB': GB, 'LR': LR, 'SVC': SVC}
string_params = {  # 'RF': ['criterion'],
                   'GB': ['loss'], 'SVC': ['kernel'],
                   'LR': ['penalty']}
param_keys = {'RF ': ['n_estimators',
                      # 'criterion',
                      'min_samples_split',
                      'min_samples_leaf'],
              'GB': ['n_estimators', 'learning_rate', 'loss', 'subsample',
                     'max_depth', 'min_samples_split', 'min_samples_leaf'],
              'SVC': ['gamma', 'C', 'kernel', 'degree'],
              'LR': ['penalty', 'C']}
param_lists = {'RF': [[5, 10, 20, 50, 100],
                      # ['gini', 'entropy'],
                      range(1, 5), range(1, 6)],
               'GB': [[5, 10, 50, 200], [10**(-i) for i in range(4)],
                      ['deviance', 'exponential'], [0.001, 0.01, 0.1, 1],
                      range(2, 7), range(1, 4), range(1, 4)],
               'SVC': [[1. / 5 * 10**(-i) for i in range(-4, 4)],
                       [10**(-i) for i in range(-4, 4)],
                       ['rbf', 'linear', 'poly', 'sigmoid'],
                       range(1, 6)
                       ],
               'LR': [['l1', 'l2'], [10**(-i) for i in range(-4, 4)]]
               }
param_space = {key: {param_keys[key][i]: param_lists[key][i] for i in
                     range(len(param_keys[key]))}
               for key in param_lists.keys()}


def mp(key, params):
    keys = param_keys[key]
    vals = param_lists[key]
    dic = {}
    for i in range(len(params)):
        # print i, keys[i]
        if keys[i] not in string_params[key]:
            dic[keys[i]] = params[i]
        else:
            # print i
            dic[keys[i]] = vals[i][params[i]]
    return dic
