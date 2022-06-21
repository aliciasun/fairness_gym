from logging import raiseExceptions
import matplotlib.pyplot as plt
import numpy as np

import torch


from graph import Graph
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.nn.functional import kl_div


class Environment(object):
    def __init__(self, name, sem, num_samples):
        self.name = name
        self.sem = sem
        self.transitions = self.sem.equations.copy()
            # self.sem.draw()
        self.states = []
        self.action_space = [0,1]
        self.reward_dic = {}
        self.current_step = 0
        self.num_samples = num_samples

        # self._init_environment()

    def sample(self, num_samples):
        sample = {}
        for v in self.topological_sort():
            if v in self.roots():
                sample[v] = self.sem.equations[v](num_samples)
            else:
                sample[v] = self.sem.equations[v](sample)
            print("DONE")
        return sample        
        

    def _init_environment(self):
        """
        init the environment by sampling from the sem
        """
        state = {}
        for v in self.sem.topological_sort():
            if v in self.sem.roots():
                state[v] = self.sem.equations[v](self.num_samples)
            elif v == 'T':
                # state[v] = torch.ones(self.num_samples)
                state[v] = torch.randint(0,2,(self.num_samples,))
            else:
                state[v] = self.sem.equations[v](state)
        self.states.append(state)


    def set_transition(self, vertex, equation):
        if vertex in self.transitions:
            del self.transitions[vertex]
            self.transitions[vertex] = equation
        else:
            self.transitions[vertex] = equation


    def update(self, current_state, action):
        """
        Update the Markov Model and return average reward
        """
        new_state = {}
        current_state['T'] = action
        for v in self.sem.topological_sort():
            if v in self.sem.roots():
                # root varibale are not updated
                new_state[v] = torch.clone(current_state[v])
            else:
                if v != 'T':
                    if v =='X':
                        # transition of features from prev state
                        new_state[v] = torch.clone(self.transitions[v](current_state))
                    else:
                        new_state[v] = torch.clone(self.transitions[v](new_state))
        # self.states.append(new_state)
        return new_state


    def get_reward(self, action):
        return NotImplementedError

    
    def compute_outcome(self, target = 'Y', agg = 'mean'):
        outcome = {}
        target_end = self.states[-1][target]
        target_init = self.states[0][target]
        Z = self.states[0]['Z']
        def dist(f,g):
            if agg == 'mean':
                return torch.mean(f-g)
            elif agg == 'kl':
                return kl_div(f,g)
            else:
                NotImplementedError
        outcome['A'] = (torch.mean(target_end[Z==0])).detach().numpy()
        outcome['B'] = (torch.mean(target_end[Z==1])).detach().numpy()
        outcome['A_change'] = dist(target_end[Z==0],target_init[Z==0]).detach().numpy()
        outcome['B_change'] = dist(target_end[Z==1],target_init[Z==1]).detach().numpy()
        # outcome['disparity_before'] =  dist(target_init[Z==1],target_init[Z==0]).detach().numpy()
        # outcome['disparity_after'] = dist(target_end[Z==1],target_end[Z==0]).detach().numpy()
        outcome['disparity_change'] = abs(outcome['A_change']-outcome['B_change'])
        return outcome
        # return torch.mean(target_var[Z==1])-torch.mean(target_var[Z==0])


    def plot_state_distribution_change(self, target = 'Y', save_path = None):
        init_var = self.states[0][target].detach().numpy()
        target_var = self.states[-1][target].detach().numpy()
        min_val = min(target_var.min(),init_var.min())
        max_val = min(target_var.max(),init_var.max())
        Z = self.states[0]['Z']
        group_0 = target_var[Z==0]
        group_1 = target_var[Z==1]
        # bins = np.linspace(min_val, max_val,50)
        bins = np.linspace(0,1)
        # plt.subplot(1, 2, 1)
        # plt.hist(init_var[Z==0], bins, density=True, alpha=0.5, label='x')
        # plt.hist(init_var[Z==1], bins, density=True, alpha=0.5, label='y')
        # plt.xlabel('Repay Probability')
        # plt.ylabel('Histogram')
        # plt.subplot(1, 2, 2)
        plt.hist(group_0, bins, weights=np.ones(len(group_0))/len(group_0), density=False,
                 alpha=0.5, label='x')
        plt.hist(group_1, bins, weights=np.ones(len(group_1))/len(group_1), density=False,
            alpha=0.5, label='y')
        plt.ylim([0, 0.5])
        plt.xlabel('Repay Probability')
        plt.legend(['group A', 'group B'],loc='upper right')
        plt.ylabel('Histogram')
        plt.savefig(save_path+'_dist_change.pdf')
        plt.close()
        # plt.show()


    def intervene(self):
        return NotImplementedError
        




class SEM(Graph):
    """
    The class instantiates a graph with equations and distributions 
    """
    def __init__(self, graph):
        super().__init__(graph)
        self.equations = {}
        self.learned = {}

    def sample(self, n_samples):
        sample = {}
        for v in self.topological_sort():
            print("Sample vertex {0}...".format(v))
            if v in self.roots():
                sample[v] = self.equations[v](n_samples)
            else:
                sample[v] = self.equations[v](sample)
            print("DONE")
        return sample


    def attach_equation(self, vertex, equation):
        """
        Attach an equation or distribution to a vertex.
        In an SEM each vertex is determined by a function of its parents (and
        independent noise), except for root vertices, which follow some
        specified distribution.
        Arguments:
            vertex: The vertex for which we attach the equation.
            equation: A callable with a single argument. For a root vertex,
            this is the number of samples to draw. For non-root vertices the
            argument is a dictionary, where the keys are the parent vertices
            and the values are torch tensors containing the data.
        """
        if vertex in self.equations:
            # print("Updating equation to vertex {}...".format(vertex), end=' ')
            del self.equations[vertex]
            self.equations[vertex] = equation
        else:
            # print("Attaching equation to vertex {}...".format(vertex), end=' ')
            self.equations[vertex] = equation
        # print("DONE")


    def intervene(self):
        pass


    def summary_equations(self):
        """
        Print a summary of the causal grpah and structural equations
        """
        import inspect
        for vertex, equation in self.equations.items():
            print("{}:{}".format(vertex,inspect.getsource(equation)))
    
    


