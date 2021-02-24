import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class State:

	def __init__(self, model, optim, criterion):
		self.model = model
		self.optim = optim
		self.criterion = criterion
		self.epoch , self.iteration = 0,0

class DotDict(dict):
	"""dot.notation access to dictionary attributes (Thomas Robert)"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

def load_yaml(path):
	with open(path, 'r') as stream:
		opt = yaml.load(stream,Loader=yaml.Loader)
	return DotDict(opt)

def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    world.dim_c = 0
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.discrete_action_space = False
    env.discrete_action_input = False
    scenario.reset_world(world)
    return env,scenario,world


def write_it(writer, cl, al, r, it):
	for i in range(len(r)):
		writer.add_scalar(f'loss/critic/agent{i}', cl[i], it)
		writer.add_scalar(f'loss/actor/agent{i}', al[i], it)
		writer.add_scalar(f'rewards/iteration/agent{i}', r[i], it)

def write_epoch(writer, mr, epoch):
	for i in range(len(mr)):
		writer.add_scalar(f'rewards/epoch/agent{i}', float(mr[i]), epoch)




