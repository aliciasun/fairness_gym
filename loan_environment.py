import torch
from fico import get_data_args
from torch.autograd import Variable
from environment import Environment, SEM


class LoanApplication(Environment):
    def __init__(self, param):
        self.num_samples = param['num_samples']
        self.num_step = param['num_steps']
        
        self.x_increase = param['x_increase']
        self.x_decrease = param['x_decrease']
        
        self.utility_repay = param['utility_repay']
        self.utility_default = param['utility_default']

        self.max_score = 850
        self.min_score = 300

        

        self.utility_ratio = self.utility_default/self.utility_repay
        self.cost_ratio = self.x_decrease/(self.x_decrease+self.x_increase)
        print("cost ratio is: {0}".format(self.cost_ratio))
        self.break_even_prob = self.utility_default/(self.utility_default+self.utility_repay)

        
  

        inv_cdfs, loan_repaid_probs, pis, group_size_ratio, scores_list, \
            rate_indices = get_data_args()
        self.inv_cdfs = inv_cdfs
        self.num_groups = len(group_size_ratio)
        self.pa = group_size_ratio[-1]   
        #p(X|Z)
        self.cdf_X_group_0, self.cdf_X_group_1 = inv_cdfs  
        #p(Y|X,Z)
        self.repay_prob_group_0, self.repay_prob_group_1= loan_repaid_probs
        self.with_noise = False
        self.root = 'real_data/cost_ratio_{:.1f}'.format(self.cost_ratio)

        self.target_variable = 'Y'
        self.utility = 0


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
                sem.attach_equation(v, lambda n: self.init_group_membership(n))
            else:
                sem.attach_equation(v, lambda n: torch.randn(n))
        sem.attach_equation("T", lambda n: self.init_action(n))
        sem.attach_equation("X", lambda state: self.init_credit_score(state))
        sem.attach_equation("Y", lambda state: self.repay_probability(state))
        sem.attach_equation("O", lambda state: self.sample_outcome(state))
        self.sem = sem
        self.transitions = self.sem.equations.copy()
        self.set_transition('X', lambda state: self.credit_score_update(state))
        self.states = []

    
    def init_group_membership(self, n):
        """
        init group distribution Z~P(Z)
        """
        z = torch.bernoulli(torch.ones(n)*self.pa)
        return Variable(z,requires_grad=True)

    def init_action(self,n):
        return torch.randint(0,2,(n,))


    def init_credit_score(self, state):
        """Sample scores X from X~P(X|Z)"""

        exogenous_noise = torch.distributions.Uniform(0., 1.).sample((self.num_samples, )).numpy()
        init_score_group_0 = self.cdf_X_group_0(exogenous_noise)
        init_score_group_1 = self.cdf_X_group_1(exogenous_noise)

        is_group_1 = (state['Z'] == torch.ones(state['Z'].shape)).float().numpy()
        output = init_score_group_1 * (is_group_1) \
                        + init_score_group_0 *(1. - is_group_1)
        if self.with_noise:
            output = output + exogenous_noise
        return Variable(torch.Tensor(output),requires_grad=True)


    def credit_score_update(self, prev_state):
        exogenous_noise = torch.rand(size=prev_state['Z'].shape).numpy()
        repay = prev_state['O']*(prev_state['T'])
        default = (1.0-prev_state['O'])*(prev_state['T'])
        output = prev_state['X'] + self.x_increase*repay - self.x_decrease*default
        output = torch.clamp(output, self.min_score, self.max_score)
        # with torch.no_grad():
        if self.with_noise:
            output = output + exogenous_noise
        return output

    
    def repay_probability(self, state, group_specific=True):
        """
        Repay probability is sampled from Y~P(Y|X,A)  
        """
        exogenous_noise = torch.rand(size=state['Z'].shape).numpy()
        is_group_1 = (state['Z'] == torch.ones(state['Z'].shape)).numpy()
        if group_specific:
            prob = self.repay_prob_group_1(state['X'].detach().numpy()) * (is_group_1) \
                    + self.repay_prob_group_0(state['X'].detach().numpy()) * (1. - is_group_1)
        else:
            prob = self.repay_prob(state['X'])
        prob = torch.clamp(torch.Tensor(prob),0,1)
        # with torch.no_grad():
        if self.with_noise:
            prob = prob + exogenous_noise
        return prob


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
    