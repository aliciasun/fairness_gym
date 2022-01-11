import matplotlib.pyplot as plt
import numpy as np

import torch


from graph import Graph


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


    def update(self, action):
        """
        Update the Markov Model and return average reward
        """
        new_state = {}
        for v in self.sem.topological_sort():
            if v in self.sem.roots():
                # root varibale are only updated at the beginning
                    new_state[v] = self.states[-1][v]
                    # new_state[v] = self.sem.equations[v](self.num_samples)

            else:
                if v == 'T':
                    #intervention node
                    new_state[v] = action
                else:
                    new_state[v] = self.transitions[v](self.states[-1])
        self.states.append(new_state)
        reward = self.get_reward(action)
        return torch.mean(reward.float())


    def get_reward(self, action):
        return NotImplementedError

    
    def avg_outcome(self, target = 'Y'):
        outcome = {}
        target_var = self.states[-1][target]
        Z = self.states[0]['Z']
        outcome['A'] = torch.mean(target_var[Z==0])
        outcome['B'] = torch.mean(target_var[Z==1])
        # return outcome
        return torch.mean(target_var[Z==1])-torch.mean(target_var[Z==0])


    def plot_state_distribution_change(self, target = 'Y', save_path = None):
        init_var = self.states[0][target].numpy()
        target_var = self.states[-1][target].numpy()
        min_val = min(target_var.min(),init_var.min())
        max_val = min(target_var.max(),init_var.max())
        Z = self.states[0]['Z']
        bins = np.linspace(min_val, max_val,50)
        plt.subplot(1, 2, 1)
        plt.hist(init_var[Z==0], bins, density=True, alpha=0.5, label='x')
        plt.hist(init_var[Z==1], bins, density=True, alpha=0.5, label='y')
        plt.subplot(1, 2, 2)
        plt.hist(target_var[Z==0], bins, density=True, alpha=0.5, label='x')
        plt.hist(target_var[Z==1], bins, density=True,alpha=0.5, label='y')
        plt.legend(['group A', 'group B'],loc='upper right')
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
    
    


