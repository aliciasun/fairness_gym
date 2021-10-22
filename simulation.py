import argparse

from environment import Environment, SEM
from agent import Agent

import torch


def simulation(env, agent, num_steps = 100):
    # action_space = [0,1]
    # agent = Agent('max_profit', action_space)
    utilities = []
    scores = []
    env._init_environment()
    env.plot_state_distribution(target='X')
    for i in range(num_steps):
        cur_state = env.states[-1]
        action = agent.get_action(cur_state)
        reward = env.update(action)
        # new_state = env.states[-1]
        # reward = env.get_reward(new_state,action)
        utilities.append(reward)
    #    print(self.states[-1]['Y'])
        scores.append(env.avg_outcome(target='X'))
    env.plot_state_distribution(target='X')
    return utilities, scores



class LoanApplication(Environment):
    def __init__(self):
        self.pa = 0.7


        self.max_score = 850
        self.min_score = 300
        self.c_increase = 1
        self.c_decrease = 1
        
        self.utility_repay = 1
        self.utility_default = 1
        self.utility = 0
        self.num_samples = 5000


        sem = SEM({
                    "NZ": None, "NX": None, "NT": None, "NY":None,
                    "Z": None,
                    "T": ["Z", "X"], 
                    "X": ["Z", "NX"], 
                    "Y": ["X", "Z", "NY"]
                    }) 

        #initialiaze
        for v in sem.roots():
            if v == 'Z':
                sem.attach_equation(v, lambda n: torch.bernoulli(torch.ones(n)*self.pa))
            else:
                sem.attach_equation(v, lambda n: torch.randn(n))
        sem.attach_equation("X", lambda state: self.init_credit_score(state))
        sem.attach_equation("Y", lambda state: self.repay_probability(state))
        self.sem = sem
        self.transitions = self.sem.equations.copy()
        self.set_transition('X', lambda state: self.credit_score_update(state))
        self.states = []

        


    def init_credit_score(self, state):
        init_score_group_0 = torch.normal(450, 25, size=state['Z'].shape)
        init_score_group_1 = torch.normal(550, 50, size=state['Z'].shape)
        is_disadvantaged = (state['Z'] == torch.zeros(state['Z'].shape)).float()
        output = init_score_group_0 ** (is_disadvantaged) \
                    * init_score_group_1 ** (1. - is_disadvantaged)+state['NX']
        return output


    def credit_score_update(self, state):
        increase = (state['Y'] >= 0.5) & (state['T']==1)
        decrease = (state['Y'] < 0.5) & (state['T']==1)
        output = state['X'] + self.c_increase*increase - self.c_decrease*decrease
        output = torch.clamp(output, self.min_score, self.max_score)
        return output

    
    def repay_probability(self, state):
        """
        Repay probability is sampled from Y~P(Y|X,A)  
        """
        return torch.bernoulli(0.001*state['X'])  


    def get_reward(self, action):
        state = self.states[-1]
        repay = (state['Y'] == 1) & ( action == 1)
        default = (state['Y'] == 0) & (action == 1)
        output = self.utility + self.utility_repay*repay - self.utility_default*default
        return output
    


if __name__ == "__main__":
    RANDOM_SEED = 42
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='loan',
                        help='name of dataset')
    args = parser.parse_args()

    if args.dataset == 'loan':
        loan_env = LoanApplication()
        utilities, scores = simulation(loan_env, num_steps=200)





