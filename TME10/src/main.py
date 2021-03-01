import matplotlib
import time
matplotlib.use("TkAgg")
import gym
import multiagent
import multiagent.scenarios
import multiagent.scenarios.simple_tag as simple_tag
import multiagent.scenarios.simple_tag as simple_spread
import multiagent.scenarios.simple_tag as simple_adversary
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from gym import wrappers, logger
import numpy as np
import copy
from utils import State, load_yaml, device, make_env, write_it, write_epoch
from ddpgMultiAgent import DDPGMultiAgent
from torch.utils.tensorboard import SummaryWriter
import argparse
import copy


def __main__():

    #---parser---#
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configuration', default='configs/simple_spread.yaml', type=str, help='choose condiguration file')
    args = parser.parse_args()

    #---config---#
    config = load_yaml(f'./{args.configuration}')

    #---env---#
    env,scenario,world = make_env(config.env)
    ob = env.reset()
    dim_state = [len(k) for k in ob]
    nber_ag = len(env.agents)
    print(f'{nber_ag} agents. Dimension of states: {dim_state}')

    #---agents---#
    agent = DDPGMultiAgent(config)

    #---writer---#
    run_id = f'{config.env}/lrcritic{config.lr_critic}_lractor{config.lr_actor}_gamma{config.gamma}_h_{config.h}_buffer{config.buffer_limit}_std{config.std}\
                _batch{config.mini_batch_size}_rho{config.rho}_ep{config.nber_ep}_act{config.nber_action}_{str(time.time()).replace(".", "_")}'
    writer = SummaryWriter(f'runs/{run_id}')

    for ep in range(config.nber_ep):

        reward = []
        verbose = (ep % config.freq_verbose == 0)
        test = ep % config.freqTest == 0
        if test:
            maxlength = config.maxLengthTest
        else:
            maxlength = config.maxLengthTrain

        for _ in range(maxlength):

            #--act--#
            a = agent.act(ob, test)             # determine action
            # print('a:', a)
            a_copy = copy.deepcopy(a)
            ob_p, r, d, i = env.step(a)     # play action
            # print('action:', a)
            # print('action_copy:', a_copy)
            # print('ob:', ob)
            # print('ob_p:', ob_p)
            # print('r:', r)
            # print('d:', d)
            # print('info:', i)

            #--store--#
            agent.store(ob, a_copy, ob_p, r, d)  # store in buffer the transition
            # print('RB:', agent.memory)
            reward.append(r)                # keep in record reward
            ob = ob_p                       # update state

            #--train--#
            if (len(agent.memory)>config.mini_batch_size) and (agent.actions % config.freq_optim == 0):
                cl, al = agent.train()
                write_it(writer, cl, al, r, agent.iteration)

            #--visualize--#
            if verbose:
                env.render(mode="none")
        mean_reward = np.array(reward).mean(axis=0)
        ob = env.reset()
        write_epoch(writer, mean_reward, ep, test)
        if ep % 10 == 0:
            print(f'Episode: {ep} - Reward: {mean_reward}')
    env.close()

if __name__ == '__main__':
    __main__()