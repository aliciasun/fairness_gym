import torch
import bisect

import numpy as np
import scipy.optimize
import scipy.spatial
from sklearn import metrics as sklearn_metrics


class ThresholdPolicy():
    """
    return action when an example has sensitive attribute z and features x
    """
    def __init__(self, state, name):
        self.type = type
        self.state = state
        self.groups = torch.unique(self.state['Z'])
        if name == 'max_profit':
            threshold = self.max_profit_threshold()
        elif name == 'eq_opportunity':
            threshold = self.eq_opp_threshold()
        else:
            NotImplementedError
        

    def get_action(self, state):
        action = torch.zeros_like(state['Z'])
        for group in self.groups:
            action[(state['Z']==group)&(state['X']>self.threshold[int(group)])]=1
        return action


    def max_profit_threshold(self):
        """
        maximize reward without constraint

        """
        return self.break_even_point


    def eq_opp_threshold(self):
        """
        Since there are many different thresholds where equality of opportunity
        constraints can hold, we simultaneously maximize reward described by a reward
        matrix."""
        roc = {}
        group_labels, group_predictions, group_weights={},{},{}
        for group in self.groups:

            group_selector = (self.state['Z']==group)
            group_weights[group] = torch.ones_like(group_selector)
            group_labels[group] = self.state['Y'][group_selector]
            group_predictions[group] = self.state['T'][group_selector]

            fprs, tprs, thresholds = sklearn_metrics.roc_curve(
            y_true=group_labels[group],
            y_score=group.predictions[group],
            sample_weight=group_weights[group])
            roc[group] = (fprs, np.nan_to_num(tprs), thresholds)

        def negative_reward(tpr_target):
            """Returns negative reward suitable for optimization by minimization."""

            my_reward = 0
            cost_matrix = np.ones(2,2)
            for group in self.groups:
                weights_ = []
                predictions_ = []
                labels_ = []
                group_selector = (self.state['Z']==group)
                for thresh_prob, threshold in _threshold_from_tpr(
                    roc[group], tpr_target, rng=None).iteritems():
                    labels_.extend(self.state['Y'][group_selector])
                    for weight, prediction in zip(group_weights[group],
                                                group_predictions[group]):
                        weights_.append(weight * thresh_prob)
                        predictions_.append(prediction >= threshold)
                confusion_matrix = sklearn_metrics.confusion_matrix(
                    labels_, predictions_, sample_weight=weights_)

            my_reward += np.multiply(confusion_matrix, cost_matrix.as_array()).sum()
            return -my_reward

        opt = scipy.optimize.minimize_scalar(
            negative_reward,
            bounds=[0, 1],
            method="bounded",
            options={"maxiter": 100})
        return ({
            group: _threshold_from_tpr(roc[group], opt.x, rng=None) for group in self.groups
        })


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
    try:
        # Add (fpr=1, tpr=0) to the convex hull to remove any points below the
        # random-performance line.
        hull = scipy.spatial.ConvexHull(np.vstack([fprs + [1], tprs + [0]]).T)
    except scipy.spatial.qhull.QhullError:
        print("Convex hull solver failed.")
        return roc
    verticies = set(hull.vertices)

    return (
        [fpr for idx, fpr in enumerate(fprs) if idx in verticies],
        [tpr for idx, tpr in enumerate(tprs) if idx in verticies],
        [thresh for idx, thresh in enumerate(thresholds) if idx in verticies],
    )


def _threshold_from_tpr(roc, tpr_target, rng):
    # First filter out points that are not on the convex hull.
    _, tpr_list, thresh_list = convex_hull_roc(roc)

    idx = bisect.bisect_left(tpr_list, tpr_target)

    # TPR target is larger than any of the TPR values in the list. In this case,
    # take the highest threshold possible.
    if idx == len(tpr_list):
        return RandomizedThreshold(
        weights=[1], values=[thresh_list[-1]], rng=rng, tpr_target=tpr_target)

    # TPR target is exactly achievable by an existing threshold. In this case,
    # do not randomize between two different thresholds. Use a single threshold
    # with probability 1.
    if tpr_list[idx] == tpr_target:
    return RandomizedThreshold(
        weights=[1], values=[thresh_list[idx]], rng=rng, tpr_target=tpr_target)

    # Interpolate between adjacent thresholds. Since we are only considering
    # points on the convex hull of the roc curve, we only need to consider
    # interpolating between pairs of adjacent points.
    alpha = _interpolate(x=tpr_target, low=tpr_list[idx - 1], high=tpr_list[idx])
    return RandomizedThreshold(
        weights=[alpha, 1 - alpha],
        values=[thresh_list[idx - 1], thresh_list[idx]],
        rng=rng,
        tpr_target=tpr_target)       
