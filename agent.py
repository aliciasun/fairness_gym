import torch
import bisect

import numpy as np
import scipy.optimize
import scipy.spatial
from sklearn import metrics as sklearn_metrics



class Agent(object):
    def __init__(self, name, action_space):
        self.actions = action_space
        self.name = name

    def get_action(self, state):
        policy = ThresholdPolicy(state, self.name)
            # threshold = {0:400, 1:500}
        action = policy.get_action(state)
        return action


    def get_total_reward():
        pass  



