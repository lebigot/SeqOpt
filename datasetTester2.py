# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 10:03:31 2016

@author: A-LAHLOU
"""
import numpy as np
from sklearn.cross_validation import StratifiedKFold
# from sklearn.cross_validation import train_test_split
# from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, log_loss
# from sklearn.metrics import accuracy_score
from sklearn.base import clone
from cleaning import cleaner
from mappings import param_lists, param_space, scores, algos, search_methods,\
    mp  # , string_params
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# from sklearn.gaussian_process.kernels import Matern
# from sklearn.preprocessing import normalize
import sys


class datasetTester:

    def __init__(self, path_to_train_set=None, algo=None,
                 search=None, score=None, logger='auto', add=True,
                 param_sp=None, param_lst=None):
        if algo is not None:
            try:
                assert algo in algos
            except AssertionError:
                print "Algorithm not defined yet"
        if algo is None:
            algo = 'RF'
        if score is not None:
            try:
                assert score in scores
            except AssertionError:
                print "Score not defined yet"
        if score is None:
            score = 'ACC'
        if search is not None:
            try:
                assert search in search_methods
            except AssertionError:
                print "Search method should be GS or RS"
        if path_to_train_set is None:
            path_to_train_set = 'data/train_titanic.csv'
        self.log_file = None
        if logger is not None and logger != sys.__stdout__:
            opening = "ab" if add else "wb"
            if logger == 'auto':
                logger = open(path_to_train_set[:-4] + ".log", opening)
            else:
                logger = open(logger, opening)
            self.log_file = logger
            print self.log_file
            self.old_stdout = sys.stdout
            sys.stdout = self.log_file
            print "Logger changed"
        self.algo = algos[algo]
        if param_sp is None:
            param_sp = param_space[algo]
        self.param_space = param_sp
        if param_lst is None:
            param_lst = param_lists[algo]
        self.param_list = param_lst
        self.mapping = lambda params: mp(algo, params)
        self.score = scores[score]
        self.search = (search_methods[search] if search else None)
        cl = cleaner(path_to_train_set, header=0)
        (self.input_data, self.identifiers, self.output) = cl.clean()

    def crossValScore(self, n_folds=3, shuffle=True,
                      return_std=False, one_vs_all=False,
                      n_jobs_1VA=2, **kwargs):
        if not one_vs_all:
            classifier_original = self.algo(**kwargs)
        else:
            from sklearn.multiclass import OneVsRestClassifier
            classifier_original = OneVsRestClassifier(self.algo(**kwargs),
                                                      n_jobs=n_jobs_1VA)
        skf = StratifiedKFold(self.output, n_folds=n_folds, shuffle=shuffle)
        scores = []
        for train_idx, test_idx in skf:
            X_train, X_test = (self.input_data.iloc[train_idx],
                               self.input_data.iloc[test_idx])
            y_train, y_test = (self.output.iloc[train_idx],
                               self.output.iloc[test_idx])
            classifier = clone(classifier_original)
            classifier.fit(X_train, y_train)
            if self.score in {roc_auc_score, log_loss}:
                y_pred = classifier.predict_proba(X_test)
            else:
                y_pred = classifier.predict(X_test)
            scores.append(self.score(y_test, y_pred))
        if return_std:
            return (np.mean(scores) * (-1 if self.score == log_loss else 1),
                    np.std(scores))
        else:
            return np.mean(scores) * (-1 if self.score == log_loss else 1)

    def scoreToOptimize(self, cv, params, loggified=[False]*len(params),
                        floatified=[False]*len(params), one_vs_all=False,
                        n_jobs_1VA=2, shuffle=True):
        if loggified:
            params = [10.**param for param in params]
        new_params = []
        for i in range(len(params)):
            temp = params[i]
            if loggified[i]:
                temp = 10. ** temp
            if floatified[i]:
                temp = int(temp)
            new_params.append(temp)
        return self.crossValScore(n_folds=cv, shuffle=shuffle,
                                  one_vs_all=one_vs_all,
                                  n_jobs_1VA=n_jobs_1VA,
                                  **self.mapping(new_params))

    def optimize(self, print_time=False, random_inits=5,
                 n_restarts_optimizer=5, criterion='EI',
                 stopping_criterion="n_iterations", max_iter=50,
                 kernel=C(1.0, (1e-3, 1e3)) * RBF([10, 1], (1e-5, 1e5)),
                 alpha=1e-10, loggified=[False]*len(params),
                 floatified=[False]*len(params), cv=3,
                 shuffle=True, one_vs_all=False,
                 n_jobs_1VA=3, **kwargs):
        print "Starting"
        from myGP4 import functionMaximizer, utils
        domain = utils.construct_domain_from_param_list(self.param_list)
        print "Printing the domain"
        print domain
        print "done\n\n\n"
        # BECAUSE OF THE LOG TRANSFORMATION, params should change!

        def to_maximize(params):
            return self.scoreToOptimize(cv=cv, params=params,
                                        loggified=loggified,
                                        floatified=floatified,
                                        one_vs_all=one_vs_all,
                                        n_jobs_1VA=n_jobs_1VA,
                                        shuffle=shuffle)
        # to_maximize = lambda params: self.scoreToOptimize(cv=cv,
        # params=params, loggified=loggified)
        fm = functionMaximizer(to_maximize,
                               domain=domain, kernel=kernel,
                               alpha=alpha,
                               n_restarts_optimizer=n_restarts_optimizer,
                               random_inits=random_inits,
                               criterion=criterion,
                               stopping_criterion=stopping_criterion,
                               max_iter=max_iter)
        print "CREATION DONE\n\n\n"
        fm.go(plot=False, stock_1D=False, print_time=print_time)
        if self.log_file is not None:
            sys.stdout = self.old_stdout
        return fm


if __name__ == '__main__':
    # old_stdout = sys.stdout
    # d = datasetTester(algo="SVC",
    #                  param_lst=[np.linspace(-4, 4, 41),
    #                             np.linspace(-4, 4, 41)],
    #                  logger=None)
    # d = datasetTester(path_to_train_set='data/train_mnist.csv',
    #                  algo="SVC",
    #                  param_lst=[np.linspace(-4, 4, 41),
    #                             np.linspace(-4, 4, 41)],
    #                  logger=None)
    d = datasetTester(path_to_train_set='data/train_titanic.csv',
                      algo='RF',
                      # basically loggify, take the min and the max, then take as many as wanted (30 each)
                      param_lst=[np.linspace(*tuple(np.log10(range(1, 100))[0, -1]), num=30),
                                 np.linspace(*tuple(np.log10(range(1, 10))[0, -1]), num=30),
                                 np.linspace(*tuple(np.log10(range(1, 10))[0, -1]), num=30)],
                      logger=None)
    print "STARTING"
    print d.scoreToOptimize(cv=2, params=[10, 10], one_vs_all=True,
                            n_jobs_1VA=2)
    fm = d.optimize(random_inits=1, max_iter=5, loggified=[True, True, True],
                    floatified=[True, True, True])
    
    print fm.g.other_info_to_stock
    # d.log_file.close()
    # sys.stdout = old_stdout

    # from matplotlib import pyplot as plt
    # plt.plot(fm.g.y)
