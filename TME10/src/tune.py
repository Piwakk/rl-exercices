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
import optuna
import copy

c = 'configs/simple_spread.yaml'
config = load_yaml(f'./{c}') 

def define_trial(trial, config):
    config.lr_critic = trial.suggest_loguniform('lr_critic', 1e-6, 1.)
    config.lr_actor = trial.suggest_loguniform('lr_actor', 1e-6, 1.)
    config.mini_batch_size = trial.suggest_int('mini_batch_size', 100, 1000)
    config.std = trial.suggest_uniform('std', 0., 0.5)
    config.rho = trial.suggest_uniform('rho', 1., 1.)
    config.h = trial.suggest_int('h', 50, 250)
    agent = DDPGMultiAgent(config)

    return agent

def objective(trial):
    global env
    global config

    agent = define_trial(trial, config)

    ob = env.reset()
    r_test = 0

    for ep in range(config.nber_ep):

        test = ep % config.freqTest == 0 and ep > 0
        max_length = config.maxLengthTest if test else config.maxLengthTrain
        reward = []

        for _ in range(max_length):

            #--act--#
            a = agent.act(ob, test)         # determine action
            a_copy = copy.deepcopy(a)
            ob_p, r, d, i = env.step(a)     # play action

            #--store--#
            agent.store(ob, a_copy, ob_p, r, d)  # store in buffer the transition
            reward.append(r)                # keep in record reward
            ob = ob_p                       # update state

            #--train--#
            if (len(agent.memory)>config.mini_batch_size) and (agent.actions % config.freq_optim == 0):
                cl, al = agent.train()
            ob = env.reset()
        if test :
            r_test = np.array(reward).mean()
            trial.report(r_test, ep)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    return r_test.mean()


env,scenario,world = make_env(config.env)
ob = env.reset()

if __name__=='__main__':

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    trial = study.best_trial
    print('value:', trial.value)
    print('Params:')
    for key, value in trial.params.items():
        print(" {}:{}".format(key, value))
    print('Best params:', study.best_params)
