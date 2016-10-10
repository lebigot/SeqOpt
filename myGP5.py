# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 14:42:39 2016

@author: A-LAHLOU
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from math import pi
from numpy import sin, cos, log, sqrt
import time
from gpr import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared
from sklearn.gaussian_process.kernels import ConstantKernel as C
# from sklearn.gaussian_process.kernels import Matern


class utils:

    @staticmethod
    def isPositiveDefinite(mat):
        n, m = np.shape(mat)
        for i in range(n):
            for j in range(m):
                if mat[i, j] < 0 and abs(mat[i, j]) < 1e-10:
                    mat[i, j] = - mat[i, j]
        # print eig(mat)
        try:
            np.linalg.cholesky(mat)
            print "OK"
        except Exception:
            print "NO"

    @staticmethod
    def construct_domain(list_of_1D_arrays):
        """
        only works with numeric arrays
        """
        n = len(list_of_1D_arrays)
        return np.transpose(np.meshgrid(*list_of_1D_arrays)).reshape(-1, n)

    @staticmethod
    def construct_domain_all_arrays(list_of_1D_arrays, return_indices=True):
        """
        only works with numeric arrays
        """
        list_of_1D_arrays = map(np.asarray, list_of_1D_arrays)
        dtypes = [str(x.dtype)[0] for x in list_of_1D_arrays]
        non_numeric_arrays_indices = [i for i in range(len(list_of_1D_arrays))
                                      if dtypes[i] not in ['i', 'f']]
        numeric_arrays = [list_of_1D_arrays[i] if i not in
                          non_numeric_arrays_indices else
                          np.arange(len(list_of_1D_arrays[i])) for i in
                          range(len(list_of_1D_arrays))]
        domain_numeric = utils.construct_domain(numeric_arrays)
        real_domain = map(lambda element: [element[i] if i not in
                                           non_numeric_arrays_indices else
                                           list_of_1D_arrays[i][element[i]]
                                           for i in range(len(element))],
                          domain_numeric)
        if return_indices:
            return (np.asarray(real_domain), dtypes)
        else:
            return np.asarray(real_domain)

    @staticmethod
    def construct_domain_kwargs(dict_of_1D_arrays):
        vals = dict_of_1D_arrays.values()
        keys = dict_of_1D_arrays.keys()
        domain_list, dtypes = utils.construct_domain_all_arrays(vals)
        # print "domain list"
        # print domain_list
        to_return = []
        for element in domain_list:
            d = {keys[i]: (element[i] if dtypes[i] not in ['i', 'f']
                           else int(element[i]) if dtypes[i] == 'i'
                           else float(element[i]))
                 for i in range(len(element))}
            to_return.append(d)
        return np.asarray(to_return)

    @staticmethod
    def construct_domain_from_param_list(param_list):
        arrays = [np.asarray(l) for l in param_list]
        for i in range(len(arrays)):
            if str(arrays[i].dtype)[0] not in {'i', 'f'}:
                # categorical variables
                arrays[i] = np.arange(len(arrays[i]))
        return utils.construct_domain(arrays)

    @staticmethod
    def repairMatrix(mat):
        val, vec = np.linalg.eig(mat)
        if (not all(np.isreal(val))) and (not all(val > 0)):
            repaired_val = np.absolute(val)
            return np.dot(np.dot(vec, np.diag(repaired_val)),
                          np.transpose(vec))
        return mat

    @staticmethod
    def plot3d_fct(f):
        from pylab import meshgrid, cm
        # from matplotlib.ticker import LinearLocator, FormatStrFormatter

        Xx = [[], []]
        Xx[0] = np.linspace(-4.5, 4.5, 91)
        Xx[1] = np.linspace(-4.5, 4.5, 91)
        X, Y = meshgrid(*Xx)
        Z = f([X, Y])[0]
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=cm.RdBu, linewidth=0, antialiased=False)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


class regMax:

    possible_external_criteria = {"EI",
                                  "PI",
                                  "UCB",
                                  "variance"}

    def __init__(self, f, domain=None, kernel=None, alpha=1e-10,
                 n_restarts_optimizer=2, random_inits=2, **kwargs):
        if kernel is None:
            kernel = kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + C(
                1e-2, (1e-3, 1e3)) * ExpSineSquared(1.0, 1.0, (1e-5, 1e5),
                                                    (1e-5, 1e5))
            # set default kernel
            # cov_amplitude = C(1.0, (1e-3, 1e3))
            # matern = Matern(length_scale=)
        if domain is None:
            domain = np.atleast_2d(np.linspace(0.01, 10, 1000)).T
        self.domain = domain
        self.f = f
        self.n_restarts_optimizer = n_restarts_optimizer
        nrs = n_restarts_optimizer
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha,
                                           n_restarts_optimizer=nrs,
                                           **kwargs)
        self.X_ind = np.random.randint(0, len(self.domain), random_inits)
        self.X = np.atleast_2d(self.domain[self.X_ind])
        # self.y = f(self.X).ravel()
        self.y = np.asarray([f(x) for x in self.X])
        self.fig_stocker = []
        self.fig_info_stocker = []
        self.current_argmax = self.domain[np.argmax(self.y)]
        self.current_max = max(self.y)
        self.current_step = random_inits
        self.other_info_to_stock = {
            'theta_after_fit_log': [], 'kernel_hyperparams_after_fit': []}
        self.gp.fit(self.X, self.y)
        self.other_info_to_stock['theta_after_fit_log'].append(
            self.gp.kernel_.theta)
        self.other_info_to_stock['kernel_hyperparams_after_fit'].append(
            self.gp.kernel_.get_params())
        self.y_pred, self.sigma = self.gp.predict(self.domain, return_std=True)
        self.current_argmax = self.domain[
            self.X_ind[np.argmax(self.y_pred[self.X_ind])]]
        self.current_max = max(self.y_pred[self.X_ind])

    def acq(self, y_pred, sigma, step):
        pass

    def acq_external(self, acq, x_ind, y_pred, sigma, step):
        assert acq in regMax.possible_external_criteria
        if acq == 'EI':
            std = sqrt(sigma[x_ind])
            diff = self.current_max - y_pred[x_ind]
            last_part = (1 - norm.cdf(diff / std))
            return std * norm.pdf(diff / std) - diff * last_part
        if acq == 'PI':
            # TODO: ?
            pass
        if acq == 'UCB':
            delta = 0.01

            def beta(t):
                return 2 * log(2 * pi ** 2 * t ** 2 / (3 * delta))
            return y_pred[x_ind] + sqrt(beta(step) * sigma[x_ind])
        if acq == 'variance':
            return sigma[x_ind]

    def pickNext(self, step):
        """
        returns the index
        """
        # TODO: TO CHANGE MAYBE ?
        acq_function = self.acq(self.y_pred, self.sigma, step)
        return np.argmax([acq_function(x_ind) if x_ind not in self.X_ind
                          else -np.inf for x_ind in range(len(self.domain))])

    def plot_1D(self, plot=True, plot_real_function=True, with_acq=True):
        x = self.domain
        f = self.f
        gp = self.gp
        y_pred = self.y_pred
        sigma = self.sigma
        if plot:
            fig = plt.figure()
            if plot_real_function:
                plt.plot(x, f(x), 'r:', label=u'$f(x)$')
            plt.plot(gp.X_train_, gp.y_train_, 'r.',
                     markersize=10, label=u'Observations')
            plt.plot(x, y_pred, 'b-', label=u'Prediction')
        if with_acq:
            acq_function = self.acq(self.y_pred, self.sigma, self.current_step)
            acq_array = np.array([acq_function(x_ind) if x_ind
                                  not in self.X_ind else -np.inf for x_ind
                                  in range(len(self.domain))])
            M = max(acq_array)
            m = max(0, min(acq_array))
            acq_array_scaled = 3 * \
                max(1, abs(self.current_max)) / (M - m + 1e-5) * acq_array
            if plot:
                message = u'Scaled acquisition. Next sample:{0}'
                label = message.format(self.domain[np.argmax(acq_array)])
                plt.plot(x, acq_array_scaled, 'g:', label=label)
        if plot:
            plt.fill(np.concatenate([x, x[::-1]]),
                     np.concatenate([y_pred - 1.9600 * sigma,
                                     (y_pred + 1.9600 * sigma)[::-1]]),
                     alpha=.5, fc='b', ec='None',
                     label='95% confidence interval')
            plt.xlabel('$x$')
            plt.ylabel('$f(x)$')
            plt.ylim(-15, 20)
            plt.legend(loc='upper right')
            self.fig_stocker.append(fig)
        plots = []
        if plot_real_function:
            plots.append([[x, f(x), 'r:'], {'label': u'$f(x)$'}])
        plots.append([[gp.X_train_, gp.y_train_, 'r.'], {
                     'markersize': 10, 'label': u'Observations'}])
        plots.append([[x, y_pred, 'b-'], {'label': u'Prediction'}])
        if with_acq:
            plots.append([[x, acq_array_scaled, 'g:'], {
                         'label': u'Scaled acquisition.'}])
        y_sigma = [y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]
        plots.append([[np.concatenate([x, x[::-1]]), np.concatenate(y_sigma)],
                      {'alpha': .5, 'fc': 'c', 'ec': 'None',
                       'label': '95% confidence interval'}])
        self.fig_info_stocker.append(plots)

    def plot_info(self, every=1, ylim=(-15, 15)):
        relevant = self.fig_info_stocker[::every]
        n_subplots = len(relevant)
        n_rows = int(sqrt(n_subplots))
        n_cols = int(n_subplots / float(n_rows) + 1)
        fig, axes = plt.subplots(n_rows, n_cols)
        for k in range(n_subplots):
            i, j = np.unravel_index(k, (n_rows, n_cols))
            subplt = axes[i, j]
            for l in range(len(relevant[k]) - 1):
                subplt.plot(*relevant[k][l][0], **relevant[k][l][1])
            subplt.fill(*relevant[k][len(relevant[k]) - 1]
                        [0], **relevant[k][len(relevant[k]) - 1][1])
            subplt.set_ylim(ylim)
            # subplt.set_xlim(0, 12)
            subplt.set_title("Step {0}".format(k * every + 1))
        subplt.set_zorder(100)
        subplt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def iteronce(self, x_ind=None):
        """
        if x_ind is not given, then it's gonna be the argmax of the acquisition
        function
        """
        if x_ind is None:
            x_ind = self.pickNext(self.current_step)
        # print "Sampling the {0}th point, corresponding to {1}".format(x_ind,
        # self.domain[x_ind])
        self.X_ind = np.append(self.X_ind, x_ind)
        x = self.domain[x_ind]
        y = self.f(x)
        # print y

        self.X = np.vstack([self.X, x])
        self.y = np.append(self.y, y)
        self.gp.fit(self.X, self.y)
        self.other_info_to_stock['theta_after_fit_log'].append(
            self.gp.kernel_.theta)
        self.other_info_to_stock['kernel_hyperparams_after_fit'].append(
            self.gp.kernel_.get_params(deep=False))
        self.y_pred, self.sigma = self.gp.predict(self.domain, return_std=True)
        if self.y_pred[x_ind] > self.current_max:
            self.current_max = self.y_pred[x_ind]
            self.current_argmax = x
        self.current_step += 1
        return x, y, self.current_max

    def process(self, n):
        for i in range(1, n + 1):
            self.iteronce()

    def getLast(self):
        return (len(self.X) - 1, self.X[-1], self.y[-1])

    def getMax_among_sampled(self):
        y_array = self.y
        ind = np.argmax(y_array)
        return (ind, self.X[ind], self.y[ind])

    def getMax_from_last_mean(self):
        last_mean = self.y_pred
        ind = np.argmax(last_mean)
        return (ind, self.domain[ind], self.y_pred[ind])


class varianceRegressor(regMax):

    def acq(self, y_pred, sigma, step):
        return lambda x_ind: sigma[x_ind]


class gpUCBRegressor(regMax):

    def __init__(self, f, domain=None, kernel=None, alpha=1e-6,
                 n_restarts_optimizer=0, random_inits=2, beta=None, **kwargs):
        regMax.__init__(self, f, domain=None, kernel=None,
                        alpha=1e-6, n_restarts_optimizer=0,
                        random_inits=2, **kwargs)
        if beta is None:
            delta = 0.01

            def beta(t):
                return 2 * log(2 * pi ** 2 * t ** 2 / (3 * delta))
        self.beta = beta

    def acq(self, y_pred, sigma, step):
        def f(x_ind):
            return y_pred[x_ind] + sqrt(self.beta(step) * sigma[x_ind] ** 2)
        return f


class eiRegressor(regMax):

    def acq(self, y_pred, sigma, step):
        def f(x_ind):
            part_m_1 = (self.current_max - y_pred[x_ind])
            part_0 = norm.pdf(part_m_1 / sigma[x_ind])
            part_1 = sigma[x_ind] * part_0
            part_2 = (self.current_max - y_pred[x_ind])
            part_2_5 = (self.current_max - y_pred[x_ind])
            part_3 = (1 - norm.cdf(part_2_5 / sigma[x_ind]))
            return part_1 - part_2 * part_3
        return f


class piRegressor(regMax):

    def acq(self, y_pred, sigma, step):
        def f(x_ind):
            part_1 = (self.current_max - y_pred[x_ind])
            return 1 - norm.cdf(part_1 / sigma[x_ind])
        return f


class randomRegressor(regMax):

    pass


class functionMaximizer:

    # add more criteria here maybe ??
    possible_criteria = {"EI": eiRegressor,
                         "PI": piRegressor,
                         "UCB": gpUCBRegressor,
                         "variance": varianceRegressor,
                         "random": randomRegressor}
    possible_stopping_criteria = {"EI", "PI", "UCB",
                                  "variance", "n_iterations", "wanted_value"}

    def __init__(self, f, domain=None, kernel=None, alpha=1e-5,
                 n_restarts_optimizer=5, random_inits=2, criterion="EI",
                 stopping_criterion=None,
                 thresh_acq=1e-3, max_iter=10, **kwargs):
        """
        Basically, one has to choose the acquisition function using one of
        the possible criteria. If random, this should be equivalent to a
        random search. One should also specify the stopping criterion.
        For example, one migh to choose EI as acquisition function, but
        stop when the maximal PI is lower than a threshold.
        In this example, stopping_criterion shoud be set to PI, and thresh_acq
        should be the threshold for PI (and not for EI !). If the stopping
        criterion choosen is n_iterations, then one should specify max_iter,
        and thresh_acq will be useless. max_iter should always be specified,
        even when the stopping criterion is not n_iterations. For example,
        f the stopping criterion is variance, this means that looping will
        stop when the maximum variance is lower than the specified threshold
        or, if the maximum number of iterations is reached. n_iterations
        should be lower than then the length of the domain kwargs should
        specify the arguments of the gaussian process used
        (e.g. kernel, nugget...)
        """
        try:
            assert criterion in functionMaximizer.possible_criteria
        except AssertionError:
            message = "criterion should be in"
            print message + str(functionMaximizer.possible_criteria)
        if stopping_criterion is None:
            stopping_criterion = criterion
        try:
            temp = functionMaximizer.possible_stopping_criteria
            assert stopping_criterion in temp
        except AssertionError:
            message = "stopping criterion should be in"
            print message + str(functionMaximizer.possible_stopping_criteria)
        temp = functionMaximizer.possible_criteria[criterion]
        self.g = temp(f, domain, kernel, alpha, n_restarts_optimizer,
                      random_inits, **kwargs)
        self.criterion = criterion
        try:
            cond_1 = max_iter is not None
            cond_2 = max_iter <= len(self.g.domain) and max_iter > 2
            assert cond_1 and cond_2
        except AssertionError:
            message = """max_iter should be specified, and should be
                         lower than the length of the domain ({0})"""
            print message.format(len(self.g.domain))
        self.stopping_criterion = stopping_criterion
        self.thresh_acq = thresh_acq
        self.max_iter = max_iter
        print "Initialization fo the fm done (in myGP4)"

    def go(self, plot=True, stock_1D=True, actual_max=None, print_time=True):
        """
        pass the actual max only when all values are nonnegative
        (e.g. in [0,1]) and of course, only if available
        """
        max_iter = self.max_iter
        thresh_acq = self.thresh_acq
        g = self.g
        message = """The random initializations are {0},
                     which led to the scores {1}"""
        print message.format(str(g.X), str(g.y))
        while True:
            message = "\nCurrently processing the {0} th iteration"
            print message.format(g.current_step)
            t = time.time()
            x, y, current_max = g.iteronce()
            message = """Sampled the function at {0}. The value is {1}.
                         The current maximum is {2}"""
            print message.format(x, y, current_max)
            if print_time:
                print "This took {0} seconds".format(time.time() - t)
            if actual_max is not None:
                message = "Up until now, we reached {0}% of the actual maximum"
                print message.format(current_max / actual_max * 100)
            if stock_1D:
                g.plot_1D(plot=plot)
            if self.criterion != 'random':
                t = time.time()
                g_acq_function = g.acq(g.y_pred, g.sigma, g.current_step)
                max_acq = max([g_acq_function(x_ind)
                               for x_ind in range(len(g.domain))])
                message = """Currently, the maximum value of the {0}
                           function is {1}"""
                print message.format(self.criterion, max_acq)
                if print_time:
                    message = "Printing this required {0} seconds"
                    print message.format(time.time() - t)
            t = time.time()
            max_ext_acq = 0
            if self.stopping_criterion != 'n_iterations':
                if self.stopping_criterion == self.criterion:
                    max_ext_acq = max_acq
                # TODO : this else bloc should be optimized !
                # In the same way that the acq
                # function was created - create the max_ext_acq function
                else:
                    if self.stopping_criterion != 'wanted_value':
                        temp = [g.acq_external(self.stopping_criterion, x_ind,
                                               g.y_pred, g.sigma,
                                               g.current_step)
                                for x_ind in range(len(g.domain))]
                        max_ext_acq = max(temp)
                message = """Currently, the maximum value of the {0} function is {1}.
                             The process will stop when it'll be lower than
                             the fixed threshold {2}"""
                print message.format(self.stopping_criterion,
                                     max_ext_acq, thresh_acq)
                if print_time:
                    message = "Printing this required {0} seconds"
                    print message.format(time.time() - t)
            stop_condition = (g.current_step > max_iter) or \
                             (self.stopping_criterion not in ['n_iterations',
                                                              'wanted_value']
                              and (max_ext_acq < thresh_acq)) or \
                             (self.stopping_criterion == 'wanted_value' and
                              (current_max > thresh_acq))

            if stop_condition:
                print "\n\nTerminating !"
                break
        return g.getMax_among_sampled(), g.getMax_from_last_mean()


if __name__ == '__main__':

    import sys
    old_stdout = sys.stdout
    log_file = open("function_maximizer_log.log", "wb")
    sys.stdout = log_file

    def f(x):
        return x ** 1.5 * sin(pi / 20 * cos(pi / 2. * x))
        # return x ** 1.5 * sin(pi * cos(pi / 2. * x))
    Xx = np.atleast_2d(np.linspace(0.01, 10, 1000)).T
    fct_name = "x -> x ** 1.5 * sin(pi/20 * cos(pi / 2. * x))"
    print "Maximizing the function {0}".format(fct_name)
    # plt.plot(Xx.ravel(), f(Xx).ravel())
    fm = functionMaximizer(f, domain=Xx,
                           kernel=C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)),
                           alpha=1e-5, n_restarts_optimizer=4, random_inits=3,
                           criterion="EI", stopping_criterion="n_iterations",
                           max_iter=10)
    max_among_sampled, max_from_last_mean = fm.go(
        plot=False, actual_max=max(f(Xx).ravel()))

    # fm.g.plot_1D()
    fm.g.plot_info(ylim=(-15, 20))
    print """\n\n Max among sampled points (with variance) is {0}.
             Max from last mean is {1}""".format(max_among_sampled,
                                                 max_from_last_mean)
    print "\n\nPrinting other info : "
    print fm.g.other_info_to_stock
    sys.stdout = old_stdout
    log_file.close()

    STOP

    import sys
    old_stdout = sys.stdout
    log_file = open("function_maximizer_log.log", "wb")
    sys.stdout = log_file

    def f(x):
        part_1 = sin(3 * pi * x[0]) ** 2 + (x[0] - 1) ** 2
        part_2 = (1 + sin(2 * pi * x[1]) ** 2) + (x[1] - 1) ** 2
        part_3 = (1 + sin(2 * pi * x[1]) ** 2)
        actual_function = np.array([part_1 * part_2 * part_3])
        return 7 - 1 / 8. * actual_function
    # print "Maximizing Beale's function"
    # def f(x):
        # return -1./500 * np.array([(x[0] + 2 * x[1] - 7) ** 2 + (2*x[0] +
        # x[1] - 5) ** 2])
    print "Maximizing Levi's function N13"
    # utils.plot3d_fct(f)
    Xx = utils.construct_domain([np.linspace(-4, 4, 201)] * 2)
    fm = functionMaximizer(f, domain=Xx, max_iter=10,
                           stopping_criterion="n_iterations",
                           **{'n_restarts_optimizer': 5})
    fm.go(plot=False, stock_1D=False)
    sys.stdout = old_stdout
    log_file.close()

    STOP

    max_iter = 70
    length_trials = 100

    random_indices = np.random.choice(range(len(Xx)), max_iter)
    X_random = [Xx[ind] for ind in random_indices]
    f_random = [f(x)[0] for x in X_random]
    max_f_random = max(f_random)
    max_f_random

    STOP

    # COMPARE SEQOPT AND RANDOM
    max_iter = 70
    length_trials = 100
    # SOME KIND OF PLOT WITH X IN 0..100
    # where Y is the expected number of iterations
    # required to reach X% of the actual maximum
    actual_max = 7.
    trials_all_seqopt = []
    trials_all_random = []
    for p in range(70, 102, 2):
        # build a trial list with 10 values corresponding
        # to the number of iterations required
        # to reach p% of the max (7) -> with a maximum of 50 iterations
        trials_seqopt = []
        trials_random = []
        for i in range(length_trials):
            print "p = {1}, i = {0}".format(i, p)
            wanted_value = p / 100. * actual_max
            fm = functionMaximizer(f, domain=Xx, max_iter=max_iter,
                                   stopping_criterion="wanted_value",
                                   thresh_acq=wanted_value)
            fm.go(plot=False, stock_1D=False)
            try:
                first_index = [i for i, j in enumerate(
                    fm.g.y) if j > wanted_value][0]
            except Exception:
                first_index = -1
            trials_seqopt.append(first_index)

            random_indices = np.random.choice(range(len(Xx)), max_iter)
            X_random = [Xx[ind] for ind in random_indices]
            f_random = [f(x)[0] for x in X_random]
            try:
                first_index = [i for i, j in enumerate(
                    f_random) if j > wanted_value][0]
            except Exception:
                first_index = -1
            trials_random.append(first_index)

        print trials_random, trials_seqopt
        trials_all_random.append(trials_random)
        trials_all_seqopt.append(trials_seqopt)
    dict_to_stock = {'random': trials_all_random, 'seqopt': trials_all_seqopt}
    import cPickle
    cPickle.dump(dict_to_stock, open("all_trials.pkl", "wb"))

    STOP

    info_seqopt = [(np.mean(trial), np.std(trial), np.min(
        trial), np.max(trial)) for trial in trials_all_seqopt]
    info_random = [(np.mean(trial), np.std(trial), np.min(
        trial), np.max(trial)) for trial in trials_all_random]

    fig = plt.figure()
    plt.plot(range(80, 102, 2)[:-1], [info[0] for info in info_random][:-1],
             linestyle='-', marker='o', color='darkkhaki',
             label='mean iterations for Random Search')
    plt.plot(range(80, 102, 2)[:-1], [info[0] for info in info_seqopt]
             [:-1], 'c-o', label='mean iterations for SeqOpt')
    plt.legend()
    plt.xlabel('Percentage of the actual maximum')
    plt.ylabel('Mean number of required iterations (over 10 trials)')
    plt.title("""Comparison of RS and SeqOpt for the task of
                 approaching the maximum of a 2D function (Levi N13)""")

    STOP

    def mix_two_lists(a, b):
        c = []
        for i in range(len(a)):
            c.append(a[i])
            c.append(b[i])
        return c

    fig, ax1 = plt.subplots(figsize=(10, 6))

    bp = plt.boxplot(mix_two_lists(
        trials_all_random[:-1], trials_all_seqopt[:-1]))

    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.set_title("""Comparison of RS and SeqOpt for the task of
                     approaching the maximum of a 2D function (Levi N13)""")
    ax1.set_xlabel('Percentage of the actual maximum')
    ax1.set_ylabel('Number of required iterations')

    boxColors = ['darkkhaki', 'cyan']
    numBoxes = len(trials_all_random[:-1]) * 2
    medians = list(range(numBoxes))

    from matplotlib.patches import Polygon

    for i in range(numBoxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX, boxY))
        # Alternate between Dark Khaki and Royal Blue
        k = i % 2
        boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
        ax1.add_patch(boxPolygon)
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            plt.plot(medianX, medianY, 'k')
            medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        # plt.plot([np.average(med.get_xdata())],
        #          [np.average(trials_all_random[:-1][i/2] if i%2
        #                      else trials_all_seqopt[:-1][i/2] )],
        #         color='w', marker='*', markeredgecolor='k')
    plt.plot(range(numBoxes + 1)[1::2], [info[0] for info in info_random][:-1],
             linestyle='-', marker='o', color='darkkhaki',
             label='mean #iterations for Random Search')
    plt.plot(range(numBoxes + 1)[2::2], [info[0] for info in info_seqopt]
             [:-1], 'c-o', label='mean #iterations for SeqOpt')
    plt.legend()
    xticks = mix_two_lists(range(80, 102, 2)[:-1], range(80, 102, 2)[:-1])
    plt.xticks(range(1, numBoxes + 1), xticks)
