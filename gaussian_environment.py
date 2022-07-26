import torch
from fico import get_data_args
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

from environment import Environment, SEM
import numpy as np



class SyntheticGaussian(Environment):
    def __init__(self, param=None):
        # generate features
        self.num_samples = param['num_samples']
        self.num_features = param['num_features']
        self.num_step = param['num_steps']

        #set environment parameters
        self.x_increase = torch.Tensor(param['x_increase'])
        self.x_decrease = torch.Tensor(param['x_decrease'])
        self.x_weight = torch.Tensor(param['x_weight'])
        
        self.utility_repay = param['utility_repay']
        self.utility_default = param['utility_default']

        self.num_groups = param['num_groups']
        self.pa = param['group_rate']
        self.feature_corr = False
        self.with_noise = False

        self.utility = 0
        self.cost_ratio = self.x_decrease/(self.x_decrease+self.x_increase)
        print("cost ratio is: {0}".format(self.cost_ratio))
        self.break_even_prob = self.utility_default/(self.utility_default+self.utility_repay)
        self.target_variable = 'Y'



        sem = SEM({
                    "NZ": None, "NX": None, "NT": None, "NY":None,
                    "Z": None,
                    "T": ["Z", "X"], 
                    "X": ["Z", "NX"], 
                    "Y": ["X", "NY"],
                    "O": ["Y"]
                    }) 

        #initialiaze
        for v in sem.roots():
            if v == 'Z':
                sem.attach_equation(v, lambda n: self.generate_init_group_membership(n))
            else:
                sem.attach_equation(v, lambda n: torch.randn(n))
        sem.attach_equation("T", lambda n: self.generate_init_action(n))
        sem.attach_equation("X", lambda state: self.generate_init_features(state))
        sem.attach_equation("Y", lambda state: self.generate_risk_score(state))
        sem.attach_equation("O", lambda state: self.sample_outcome(state))
        self.sem = sem
        self.transitions = self.sem.equations.copy()
        self.set_transition('X', lambda state: self.feature_update(state))
        self.states = []


    
    def generate_init_group_membership(self, n):
        """
        init group distribution Z~P(Z)
        """
        z = torch.bernoulli(torch.ones(n)*self.pa)
        return Variable(z,requires_grad=True)

    def generate_init_action(self,n):
        return torch.randint(0,2,(n,))


    def generate_init_features(self, state):
        """Sample scores X from X~P(X|Z)"""
        exogenous_noise = torch.rand(size=state['Z'].shape).numpy()
        self.mu = [torch.zeros(self.num_features),torch.ones(self.num_features)]
        if self.feature_corr:
            self.cov = [torch.ones(self.num_features),torch.ones(self.num_features)]
        else:
            # self.cov = [torch.eye(self.num_features),torch.eye(self.num_features)]
            self.cov = [torch.eye(self.num_features),torch.eye(self.num_features)]
        init_features_group_0 = MultivariateNormal(self.mu[0],self.cov[0]).sample(state['Z'].shape)
        init_features_group_1 = MultivariateNormal(self.mu[1],self.cov[1]).sample(state['Z'].shape)

        is_group_1 = (state['Z'] == torch.ones(state['Z'].shape)).float().numpy()
        output = init_features_group_1 * (is_group_1[:,None]) \
                        + init_features_group_0 *(1. - is_group_1[:,None])
        if self.with_noise:
            output = output + exogenous_noise
        return Variable(torch.Tensor(output),requires_grad=True)


    def feature_update(self, prev_state):
        exogenous_noise = torch.rand(size=prev_state['Z'].shape).numpy()
        pos_outcome = prev_state['O']*(prev_state['T'])
        neg_outcome = (1.0-prev_state['O'])*(prev_state['T'])
        output = prev_state['X'] + pos_outcome[:,None]*self.x_increase[None,:] - neg_outcome[:,None]*self.x_decrease[None,:]
        # with torch.no_grad():
        if self.with_noise:
            output = output + exogenous_noise
        return output

    
    def generate_risk_score(self, state, group_specific=True):
        """
        risk score is sampled from Y~P(Y|X,A) where sigmoid function is used
        """
        exogenous_noise = torch.rand(size=state['Z'].shape).numpy()
        weight = self.x_weight
        prob = 1./(1.+torch.exp(-1.*torch.mm(state['X'],weight[:,None])))
        if self.with_noise:
            prob = prob + exogenous_noise
        # prob = (state['X']-torch.min(state['X']))/(torch.max(state['X'],torch.min(state['X'])))
        return torch.squeeze(prob)
        


    def sample_outcome(self, state):
        p = state['Y']
        exogenous_noise = torch.distributions.Uniform(0., 1.).sample(p.shape)
        log = torch.log
        EPS = 1e-8
        log = lambda x: torch.log(torch.clamp(x, EPS, 1.))  # numer. stable log
        output = ((
            log(p) - log(1. - p) +
            log(exogenous_noise) - log(1. - exogenous_noise))
                > 0.0).int()  # Gumbel-max trick
        return output


    def break_even_rate(self, rate, state):
        # repay_prob = np.quantile(state['Y'].detach().numpy(), rate)
        return abs(rate-self.break_even_prob)


    def get_reward(self, state, action):
        repay = state['O']*action
        default = (1.-state['O'])*action
        output = torch.mean(self.utility_repay*repay - self.utility_default*default)
        self.utility += output
        return output


    def find_eq_point(self):
        x = torch.ones(1, 1, requires_grad=True) 
        state = {'Z':torch.tensor([1]),'X':x}
        output = self.generate_risk_score(state)
        output.backward()
        x_grad = state['X'].grad
        eq_point = (x_grad*self.x_decrease)/(x_grad*(self.x_decrease+self.x_increase))

        # eq_point = torch.mm(x_grad,torch.Tensor(self.x_decrease))/torch.mm(x_grad,torch.Tensor(self.x_decrease+self.x_increase))
        # print(eq_point)
        return eq_point

    