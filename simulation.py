import argparse
from tqdm import tqdm
from environment import Environment, SEM
from agent import Agent

import torch
from fico import get_data_args

torch.manual_seed(42)

def simulation(env, agent, num_steps = 100):
    utilities = []
    scores = []
    env._init_environment()
    for i in tqdm(range(num_steps)):
        cur_state = env.states[-1]
        # print("X: {0}, Y:{1}, T: {2}".format(cur_state['X'],cur_state['Y'],cur_state['T']))
        action = agent.get_action(cur_state)
        reward = env.update(action)
        # new_state = env.states[-1]
        # reward = env.get_reward(new_state,action)
        utilities.append(reward)
        scores.append(env.avg_outcome(target='X'))
    env.plot_state_distribution_change(target='X',save_path=agent.name)
    return utilities, scores



class LoanApplication(Environment):
    def __init__(self, setting = 'synthetic'):

        self.max_score = 850
        self.min_score = 300
        self.c_increase = 75
        self.c_decrease = 75
        
        self.utility_repay = 50
        self.utility_default = 50
        self.break_even_prob = self.utility_default/(self.utility_default+self.utility_repay)
        self.utility = 0

        self.num_samples = 1000
        self.setting = setting

        if setting == 'synthetic':
            self.pa = 0.3
            self.num_groups = 2
        else:
            inv_cdfs, loan_repaid_probs, pis, group_size_ratio, scores_list, \
                rate_indices = get_data_args()
            #p(Z=1)
            self.inv_cdfs = inv_cdfs
            self.num_groups = len(group_size_ratio)
            self.pa = group_size_ratio[-1]   
            #p(X|Z)
            self.cdf_X_group_0, self.cdf_X_group_1 = inv_cdfs  
            #p(Y|X,Z)
            self.repay_prob_group_0, self.repay_prob_group_1 = loan_repaid_probs




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
        """Sample scores X from X~P(X|A)"""
        if self.setting == 'synthetic':
            init_score_group_0 = torch.normal(450, 25, size=state['Z'].shape)
            init_score_group_1 = torch.normal(550, 50, size=state['Z'].shape)
        else:
            exogenous_noise = torch.rand(size=state['Z'].shape).numpy()
            init_score_group_0 = self.cdf_X_group_0(exogenous_noise)
            init_score_group_1 = self.cdf_X_group_1(exogenous_noise)

        is_group_1 = (state['Z'] == torch.ones(state['Z'].shape)).float().numpy()
        output = init_score_group_1 ** (is_group_1) \
                    * init_score_group_0 ** (1. - is_group_1)
        return output


    def credit_score_update(self, state):
        increase = (state['Y'] == 1) & (state['T']==1)
        decrease = (state['Y'] == 0) & (state['T']==1)
        output = state['X'] + self.c_increase*increase - self.c_decrease*decrease
        output = torch.clamp(output, self.min_score, self.max_score)
        return output

    
    def repay_probability(self, state):
        """
        Repay probability is sampled from Y~P(Y|X,A)  
        """
        if self.setting == 'synthetic':
            prob = 0.001*state['X']
            output = torch.bernoulli(prob)
        else:  
            is_group_1 = (state['Z'] == torch.ones(state['Z'].shape)).numpy()
            prob = self.repay_prob_group_1(state['X']) ** (is_group_1) \
                    * self.repay_prob_group_0(state['X']) ** (1. - is_group_1)
            output = torch.bernoulli(torch.Tensor(prob))
        return output



    def get_reward(self, action):
        state = self.states[-1]
        repay = (state['Y'] == 1) & (action == 1)
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
        action_space = [0,1]
        agent_prof = Agent(loan_env, 'max_profit', action_space)
        agent_eo = Agent(loan_env, 'eq_opp', action_space)  
        util_prof, score_prof = simulation(loan_env, agent_prof, num_steps=10)

        util_eo, score_eo = simulation(loan_env, agent_eo, num_steps=10)





