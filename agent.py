import torch
from torch import nn
import bisect

import numpy as np
import scipy.optimize
import scipy.spatial
from sklearn import metrics as sklearn_metrics
from threshold_policy import ThresholdPolicy

class Agent(object):
    def __init__(self, env, name, action_space):
        self.env = env
        self.actions = action_space
        self.name = name
        self.policy = ThresholdPolicy(self.env, self.name)

    def get_action(self, state):
            # threshold = {0:400, 1:500}
        action = self.policy.get_action(state)
        return action


    def get_total_reward():
        pass  

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        self.linear = nn.Linear(input_dim, output_dim)
        pass



class LearnedPolicy(nn.Module):
    def __init__(self, name, env):
        if name == 'polynomial':
            weight = torch.parameter()
        else:
            pass
        self.name = name
        self.env = env
        # self.state = state
        self.groups = self.env.num_groups
        self.cost_matrix = np.array([[0, -self.env.utility_default],[0, self.env.utility_repay]])
        self.thresholds = [[],[]]
 


    def get_reward(thresholds, state):
        group_A = state['X'](state['Z'] == 0)
        group_B = state['X'](state['Z'] == 1)
        dist_A = torch.tensor(np.histogram(group_A)[-1])
        dist_B = torch.tensor(np.histogram(group_B)[-1])
        disparity = nn.KLDivLoss(dist_A, dist_B)


    def get_action(self, state, rng=None):
        if rng is None:
            rng = np.random.RandomState(42)
        action = np.zeros_like(state['Z'])
        for group in range(self.groups):
            threshold = thresholds[int(group)].sample()
            # print("threshold for group {0} is {1}".format(group,threshold))
            action[(state['Z']==group)&(state['X']>threshold)]=1
            self.thresholds[group].append(threshold)
        return action
        


