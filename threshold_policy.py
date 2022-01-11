import matplotlib.pyplot as plt

import torch
import bisect
import attr
import logging

import numpy as np
import scipy.optimize
import scipy.spatial
from sklearn import metrics as sklearn_metrics

from fico import get_data_args


class ThresholdPolicy():
    """
    return a threshold policy when an example has sensitive attribute z and features x
    """
    def __init__(self, env, name):
        self.name = name
        self.env = env
        # self.state = state
        self.groups = self.env.num_groups
        self.cost_matrix = np.array([[0, -self.env.utility_default],[0, self.env.utility_repay]])
        self.thresholds = [[],[]]
 
    def get_action(self, state, rng=None):
        if rng is None:
            rng = np.random.RandomState(42)
        action = np.zeros_like(state['Z'])
        if self.name == 'max_profit':
            thresholds = self.max_profit_threshold(state, rng=rng)
        elif self.name == 'eq_opp':
            thresholds = self.eq_opp_threshold(state,rng=rng)
        else:
            NotImplementedError
        for group in range(self.groups):
            threshold = thresholds[int(group)].sample()
            # print("threshold for group {0} is {1}".format(group,threshold))
            action[(state['Z']==group)&(state['X']>threshold)]=1
            self.thresholds[group].append(threshold)
        return action
    
    # def get_reward(self, state, roc, tpr_target):

    #     """Returns negative reward suitable for optimization by minimization."""
    #     reward = 0



    def get_profit(self, rate):
        # per_group_profit = self.env.
        for groups in range(self.groups):
            score = self.env.cdf_X_group_0(rate)


    def max_profit_threshold(self, state, rng=None):
        """
        maximize reward without constraint
        """
        break_even_prob = self.env.break_even_prob
        # opt = scipy.optimize.root(self.env.break_even_rate,0.5,args=state)
        opt = scipy.optimize.minimize_scalar(
                self.env.break_even_rate,
                bounds=[0, 1],
                method="bounded",
                options={"maxiter": 100},
                args=state)

        # def negative_reward(rate):
        #     threshold = np.quantile(state['X'],rate)
        #     predictions = (state['X']>=threshold)
        #     confusion_matrix = sklearn_metrics.confusion_matrix(state['Y'], predictions)
        #     reward = np.multiply(confusion_matrix, self.cost_matrix).sum()
        #     return -reward
        # opt = scipy.optimize.minimize_scalar(
        #         negative_reward,
        #         bounds=[0, 1],
        #         method="bounded",
        #         options={"maxiter": 100})
        threshold_score = np.quantile(state['X'],opt.x)
        return ({
            group: RandomizedThreshold(weights=[1], values=[threshold_score], rng=rng) for group in range(self.groups)})
        # return np.trunc(self.env.perf+1-break_even_prob)
        
    def demo_parity_threshold(self, state, rng=None):
        """
        Accept the same percentage of people regardless of group
        """
        pass

    def eq_opp_threshold(self, state, rng=None):
        num_group = len(np.unique(state['Z']))
        roc = {}
        group_weights = {}
        group_labels = {}
        group_predictions = {}

        for group in range(num_group):
            group_selector = (state['Z'] == group)
            group_predictions[group] = state['X'][group_selector]
            group_labels[group] = state['Y'][group_selector]
            group_weights[group] = [1 for _ in group_labels[group]]

            fprs, tprs, thresholds = sklearn_metrics.roc_curve(
                y_true=group_labels[group],
                y_score=group_predictions[group],
                sample_weight=group_weights[group])
            roc[group] = (fprs, np.nan_to_num(tprs), thresholds)
        def negative_reward(tpr_target):
            reward = 0
            for group in range(num_group):
                weights_ = []
                predictions_ = []
                labels_ = []
                for thresh_prob, threshold in threshold_from_tpr(roc[group], tpr_target, rng=rng).iteritems():
                    labels_.extend(group_labels[group])
                    for weight, prediction in zip(group_weights[group],
                                        group_predictions[group]):
                        weights_.append(weight * thresh_prob)
                        predictions_.append(prediction >= threshold)
                confusion_matrix = sklearn_metrics.confusion_matrix(
                    labels_, predictions_, sample_weight=weights_)
                reward += np.multiply(confusion_matrix, self.cost_matrix).sum()
            return -reward

        opt = scipy.optimize.minimize_scalar(
                negative_reward,
                bounds=[0, 1],
                method="bounded",
                options={"maxiter": 100})
        return ({
            group: threshold_from_tpr(roc[group], opt.x, rng=rng) for group in range(num_group)})


    def plot_policy(self):
        plt.plot(self.thresholds[0],linestyle='dashed',color='r')
        plt.plot(self.thresholds[1],linestyle='dashed',color='blue')
        plt.ylim([300, 850])
        plt.savefig(self.name+'_thresholds.pdf')
        plt.close()




def threshold_from_tpr(roc, tpr_target,rng):
     #find closest fpr 
    _, tpr_list, thresh_list = convex_hull_roc(roc)
    idx = bisect.bisect_left(tpr_list, tpr_target)
    if idx == len(tpr_list):
        return RandomizedThreshold(
        weights=[1], values=[thresh_list[-1]], rng=rng, tpr_target=tpr_target)
    if tpr_list[idx] == tpr_target:
        return RandomizedThreshold(
            weights=[1], values=[thresh_list[idx]], rng=rng, tpr_target=tpr_target)
    # Interpolate between adjacent thresholds. 
    alpha = _interpolate(x=tpr_target, low=tpr_list[idx - 1], high=tpr_list[idx])
    return RandomizedThreshold(
        weights=[alpha, 1 - alpha],
        values=[thresh_list[idx - 1], thresh_list[idx]],
        tpr_target=tpr_target)


def eq_opp_threshold(self, state):
    thresholds = {}
    break_even_prob = self.env.break_even_prob
    f_profit = lambda tp: self.get_profit(tp)
    tp_opt = ternary_maximize(f_profit)
    for i in len(self.groups):
        thresholds[i]= self.threshold_from_tpr(tp_opt)
    return thresholds




def demographic_parity_threshold(self):
    f_profit = lambda rate: self.get_profit(rate)
    rate_opt = ternary_maximize(f_profit)



def _interpolate(x, low, high):
  """returns a such that a*low + (1-a)*high = x."""
  assert low <= x <= high, ("x is not between [low, high]: Expected %s <= %s <="
                            " %s") % (low, x, high)
  alpha = 1 - ((x - low) / (high - low))
  assert np.abs(alpha * low + (1 - alpha) * high - x) < 1e-6
  return alpha

 


def convex_hull_roc(roc):
    """Returns an roc curve without the points inside the convex hull.
  Points below the fpr=tpr line corresponding to random performance are also
  removed.
    Args:
    roc: A tuple of lists that are all the same length, containing
      (false_positive_rates, true_positive_rates, thresholds). This is the same
      format returned by sklearn.metrics.roc_curve.
    """
    fprs, tprs, thresholds = roc
    if np.isnan(fprs).any() or np.isnan(tprs).any():
        logging.debug("Convex hull solver does not handle NaNs.")
        return roc
    if len(fprs) < 3:
        return roc
    try:
        # Add (fpr=1, tpr=0) to the convex hull to remove any points below the
        # random-performance line.
        hull = scipy.spatial.ConvexHull(np.vstack([fprs + [1], tprs + [0]]).T)
    except scipy.spatial.qhull.QhullError:
        logging.debug("Convex hull solver failed.")
        return roc
    verticies = set(hull.vertices)
    return (
        [fpr for idx, fpr in enumerate(fprs) if idx in verticies],
        [tpr for idx, tpr in enumerate(tprs) if idx in verticies],
        [thresh for idx, thresh in enumerate(thresholds) if idx in verticies],
    )





def a_max(f, lst):
    """Finds max function value in a list"""
    val_max = lst[0]
    for l in lst:
        if f(l) > f(val_max):
            val_max = l
    return val_max


def binary_search(f, target, left=1e-5, right=1 - 1e-5, tol=1e-8):
    """Binary search implementation"""
    midpoint = (left + right) * .5
    if abs(midpoint - left) < tol:
        return midpoint
    elif f(midpoint) < target:
        return binary_search(f, target, midpoint, right, tol)
    elif f(midpoint) >= target:
        return binary_search(f, target, left, midpoint, tol)
    else:
        print("Error - we should never get down here")


def ternary_maximize(f, left=1e-5, right=1 - 1e-5, tol=1e-5):
    """ternary search on the equalized criterion (loan rate, ppv, etc.)"""
    m_1 = (2. / 3) * left + (1. / 3) * right
    m_2 = (1. / 3) * left + (2. / 3) * right
    if abs(m_1 - m_2) < tol:
        return a_max(f, [m_1, m_2])
    if f(m_1) < f(m_2):
        return ternary_maximize(f, m_1, right, tol)
    if f(m_1) >= f(m_2):
        return ternary_maximize(f, left, m_2, tol)
    else:
        print("Error - we should never get down here")


@attr.s
class RandomizedThreshold(object):
    """Represents a distribution over decision thresholds."""
    values = attr.ib(factory=lambda: [0.])
    weights = attr.ib(factory=lambda: [1.])
    rng = attr.ib(factory=np.random.RandomState)
    tpr_target = attr.ib(default=None)

    def smoothed_value(self):
        # If one weight is small, this is probably an optimization artifact.
        # Snap to a single threshold.
        if len(self.weights) == 2 and min(self.weights) < 1e-4:
            return self.values[np.argmax(self.weights)]
        return np.dot(self.weights, self.values)

    def sample(self):
        return self.rng.choice(self.values, p=self.weights)

    def iteritems(self):
        return zip(self.weights, self.values)