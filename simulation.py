import argparse
import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import yaml


from environment import Environment, SEM
from loan_environment import LoanApplication
from gaussian_environment import SyntheticGaussian
from agent import Agent

import torch

torch.manual_seed(42)

def simulation(env,agent, result_dir, num_steps = 100,  plot=False):
    res = {}
    res['util'] = np.zeros((1,num_steps))
    res['disparity'] = np.zeros((1,num_steps))
    res['mean_target'] = np.zeros((2,num_steps))
    res['mean_action'] = np.zeros((2,num_steps))
    cum_reward = 0
    num_steps = 5
    for i in tqdm(range(num_steps)):

        cur_state = env.states[-1]
        # print(cur_state['Y'])
        print(torch.mean(cur_state['X']))
        print(torch.mean(cur_state['Y']))

        action = agent.get_action(env.states)
        action = torch.ones_like(env.states[0]['Z'])
        reward = env.get_reward(cur_state,action)
        cur_state['Y'] = 0.5*torch.zeros_like(env.states[0]['Z'])

        new_state = env.update(cur_state,action)
        # env.find_eq_point()
        # print(torch.mean(new_state['Y']))
        env.states.append(new_state)
        outcome = env.compute_outcome(target=env.target_variable)
        cum_reward+=reward
        res['util'][0,i]=reward
        res['mean_action'][0,i]=np.mean((cur_state['T'][cur_state['Z']==0]).detach().numpy())
        res['mean_action'][1,i]=np.mean((cur_state['T'][cur_state['Z']==1]).detach().numpy())
        res['mean_target'][0,i]=outcome['A']
        res['mean_target'][1,i]=outcome['B']
        res['disparity'][0,i]=outcome['disparity_change']


    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    # if not os.path.exists('figures/{0}'.format(env.root)):
    #     os.makedirs('figures/{0}'.format(env.root))
    np.savez(res_dir+'/'+ agent_name+'_states', env.states)
    agent.policy.save_policy(save_path = res_dir+'/'+agent.name+'_threshold')

    # save_path = os.path.join('figures',env.root,str(num_steps) +'_'+agent.name)
    # if plot:
    #     # env.plot_state_distribution_change(target='Y',save_path=save_path)
    return res






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gaussian_1d',
                        help='name of dataset')
    parser.add_argument('--setting', type=str, default='setting1',
        help='name of dataset')
    parser.add_argument('--save', type=bool, default=False,
        help='save result')
    args = parser.parse_args()

    

    action_space = [0,1]

    agent_lists = ['max_profit','demo_parity','eq_opp']
    CONFIG_PATH = os.path.join("config", args.dataset)
    config_name = args.setting+'.yaml'
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    num_steps = config['num_steps']
    res_dir = os.path.join('result', args.dataset, args.setting)
    # agent_lists = ['max_profit']
    for i in range(len(agent_lists)):
        agent_name = agent_lists[i]
        if args.dataset== 'gaussian_2d' or args.dataset == 'gaussian_1d':
            env = SyntheticGaussian(param=config)
        elif args.dataset == 'loan':
            env = LoanApplication(param=config) 
        else:
            pass
        env._init_environment()
        agent =  Agent(env, agent_name, action_space)
        res = simulation(env, agent, res_dir, num_steps=num_steps, plot=True)
        if args.save:
            save_path = os.path.join(res_dir,'{0}.npz'.format(agent_name))
            np.savez(save_path, **res)
    



        # p_l_ratio = [0.25, 0.5, 1,2, 5]
        # utility_ratio = [0.25, 0.5, 1,2, 5]
        # scores_eo = []
        # scores_prof = []
        # for util in utility_ratio:
        #     print("current util is {0}".format(util))
        #     for pl in p_l_ratio:
        #         print("current util is {0}".format(util))
        #         param = {}
        #         param['p_l_ratio'] = pl
        #         param['utility_ratio'] = util
        #         loan_env = LoanApplication(param=param)
        #         util_prof, score_prof = simulation(loan_env, agent_prof, num_steps=100)
        #         scores_prof.append(score_prof[-1])
        #         loan_env = LoanApplication(param=param)
        #         util_eo, score_eo = simulation(loan_env, agent_eo, num_steps=100)
        #         scores_eo.append(score_eo[-1])
        #         print(score_prof[-1])
        #         print(score_eo[-1])





