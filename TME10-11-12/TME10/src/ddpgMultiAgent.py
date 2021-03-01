import matplotlib

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
import torch
from torch import nn
import torch.nn.functional as F
from utils import State, load_yaml, device
from collections import deque
import random


"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
	env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

<<<<<<< HEAD
class Mu(nn.Module):
	"""docstring for MU"""
	def __init__(self, config):
		super(Mu, self).__init__()
		# self.ff = nn.Sequential(
		# 	nn.Linear(config.dim_state, config.h),
		# 	nn.BatchNorm1d(config.h),
		# 	nn.ReLU(),
		# 	nn.Linear(config.h, config.dim_action)
		# 	)
		self.ff = nn.Sequential(
			nn.Linear(config.dim_state, config.h),
			nn.ReLU(),
			nn.Linear(config.h, config.dim_action)
			)
		self.clipped = config.action_high

	def forward(self, x):
		return self.clipped * torch.tanh(self.ff(x))

class Q(nn.Module):
	"""docstring for Q"""
	def __init__(self, config):
		super().__init__()
		h = config.h
		# self.ff = nn.Sequential(
		# 	nn.Linear(config.dim_state + config.dim_action * config.nber_ag, 2 * h),
		# 	nn.BatchNorm1d(2 * h),
		# 	nn.ReLU(),
		# 	nn.Linear(2 * h, h),
		# 	nn.BatchNorm1d(h),
		# 	nn.ReLU(),
		# 	nn.Linear(h, 1)
		# 	)
		self.ff = nn.Sequential(
			nn.Linear(config.dim_state + config.dim_action * config.nber_ag, 2 * h),
			nn.ReLU(),
			nn.Linear(2 * h, h),
			nn.ReLU(),
			nn.Linear(h, 1)
			)

	def forward(self, s, a_list):
		x = torch.cat([s] + a_list, dim=-1)
		return self.ff(x)


class DDPGAgent(nn.Module):

	def __init__(self, config):
		super().__init__()
		#---parameters---#
		h = config.h
		self.dim_state = config.dim_state
		self.dim_action = config.dim_action
		self.std = config.std
		self.action_low, self.action_high = torch.tensor(config.action_low, dtype=float, device=device), \
											torch.tensor(config.action_high, dtype=float, device=device)

		#---create the actor and critic nn---#
		self.mu, self.mu_target = Mu(config), Mu(config)
		self.q, self.q_target = Q(config), Q(config)
		self.mu_target.load_state_dict(self.mu.state_dict())
		self.q_target.load_state_dict(self.q.state_dict())
		self.mu.eval()

		#---optimizers tools---#
		self.optim_q = torch.optim.Adam(params=self.q.parameters(), lr=config.lr_critic)
		self.optim_mu = torch.optim.Adam(params=self.mu.parameters(), lr=config.lr_action)

		#---memory---#
		self.memory = deque(maxlen=config.buffer_limit)

		#---device---#
		self.to(dtype=torch.float64, device=device)

	def act(self, s):
		eps = torch.empty(self.dim_action).normal_(mean=0,std=self.std)  # noise for exploration
		action = self.mu(s.unsqueeze(0)) + eps
		action = torch.clamp(action, self.action_low, self.action_high)
		return action.flatten().detach().numpy()

class DDPGMultiAgent(nn.Module):
	"""docstring for DDPGMultiAgent"""
	def __init__(self, config):
		super(DDPGMultiAgent, self).__init__()

		#--tools--#
		self.epoch , self.iteration, self.actions = 0, 0, 0

		#---parameters---#
		self.nber_ag = config.nber_ag
		self.agent_list = [ DDPGAgent(config) for _ in range(config.nber_ag) ]
		self.gamma = config.gamma
		self.rho = config.rho
=======
>>>>>>> 112bcdd1b5939d017120bcdf90ee50bcf823897a

class Mu(nn.Module):
    """docstring for MU"""

    def __init__(self, config):
        super(Mu, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(config.dim_state, config.h),
            nn.BatchNorm1d(config.h),
            nn.ReLU(),
            nn.Linear(config.h, config.dim_action),
        )
        self.clipped = config.action_high

<<<<<<< HEAD
			s, s_p, a, r = self.sample()
			a_p = [ agent.mu_target(s_p[j]) for j, agent in enumerate(self.agent_list) ]

			#--critic--#
			y = r[i].unsqueeze(-1) + self.gamma * agent_i.q_target(s_p[i], a_p)
			q = agent_i.q(s[i], a)
			critic_loss = F.smooth_l1_loss(q, y.detach())

			agent_i.optim_q.zero_grad()
			critic_loss.backward()
			torch.nn.utils.clip_grad_norm_(agent_i.q.parameters(), 0.5)
			agent_i.optim_q.step()

			#--actor--#
			mu = [ agent.mu(s[j]) for j, agent in enumerate(self.agent_list) ]
			actor_loss =  - agent_i.q(s[i], mu).mean(dim=0)
			actor_loss += (mu[i]**2).mean() * 1e-3

			agent_i.optim_mu.zero_grad()
			actor_loss.backward()
			torch.nn.utils.clip_grad_norm_(agent_i.mu.parameters(), 0.5)
			agent_i.optim_mu.step()
=======
    def forward(self, x):
        return self.clipped * torch.tanh(self.ff(x))


class Q(nn.Module):
    """docstring for Q"""

    def __init__(self, config):
        super().__init__()
        h = config.h
        self.ff = nn.Sequential(
            nn.Linear(config.dim_state + config.dim_action * config.nber_ag, 2 * h),
            nn.BatchNorm1d(2 * h),
            nn.ReLU(),
            nn.Linear(2 * h, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )

    def forward(self, s, a_list):
        x = torch.cat([s] + a_list, dim=-1)
        return self.ff(x)

>>>>>>> 112bcdd1b5939d017120bcdf90ee50bcf823897a

class DDPGAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ---parameters---#
        h = config.h
        self.dim_state = config.dim_state
        self.dim_action = config.dim_action
        self.std = config.std
        self.action_low, self.action_high = (
            torch.tensor(config.action_low, dtype=float, device=device),
            torch.tensor(config.action_high, dtype=float, device=device),
        )

        # ---create the actor and critic nn---#
        self.mu, self.mu_target = Mu(config), Mu(config)
        self.q, self.q_target = Q(config), Q(config)
        self.mu_target.load_state_dict(self.mu.state_dict())
        self.q_target.load_state_dict(self.q.state_dict())

        # ---optimizers tools---#
        self.optim = torch.optim.Adam(params=self.parameters(), lr=config.lr)

        # ---memory---#
        self.memory = deque(maxlen=config.buffer_limit)

        # ---device---#
        self.to(dtype=torch.float64, device=device)

    def act(self, s):
        self.mu.eval()
        eps = torch.empty(self.dim_action).normal_(
            mean=0, std=self.std
        )  # noise for exploration
        action = self.mu(s.unsqueeze(0)) + eps
        action = torch.where(
            action < self.action_low, self.action_low, action
        )  # action clipped into possible values
        action = torch.where(
            action > self.action_high, self.action_high, action
        )  # action clipped into possible values
        return action.flatten().detach().numpy()

<<<<<<< HEAD
			agent_i.mu.eval()

		for agent in self.agent_list:
			soft_update(agent.q_target, agent.q, rho=self.rho)
			soft_update(agent.mu_target, agent.mu, rho=self.rho)
=======
>>>>>>> 112bcdd1b5939d017120bcdf90ee50bcf823897a

class DDPGMultiAgent(nn.Module):
    """docstring for DDPGMultiAgent"""

    def __init__(self, config):
        super(DDPGMultiAgent, self).__init__()

        # --tools--#
        self.epoch, self.iteration, self.actions = 0, 0, 0

        # ---parameters---#
        self.nber_ag = config.nber_ag
        self.agent_list = [DDPGAgent(config) for _ in range(config.nber_ag)]
        self.gamma = config.gamma
        self.rho = config.rho

        # ---memory---#
        self.memory = deque(maxlen=config.buffer_limit)
        self.mini_batch_size = config.mini_batch_size

    def act(self, ob):
        s = phi(ob)
        a = [agent.act(s[i]) for i, agent in enumerate(self.agent_list)]
        self.actions += 1
        return a

    def store(self, ob, a, ob_p, r, d):
        self.memory.append([phi(ob), phi(ob_p), a, r])

    def sample(self):
        mini_batch = random.sample(self.memory, self.mini_batch_size)
        s, s_prime, actions, reward = (
            [
                torch.stack([e[0][i] for e in mini_batch]).to(device)
                for i in range(self.nber_ag)
            ],
            [
                torch.stack([e[1][i] for e in mini_batch]).to(device)
                for i in range(self.nber_ag)
            ],
            [
                torch.tensor([e[2][i] for e in mini_batch], device=device, dtype=float)
                for i in range(self.nber_ag)
            ],
            [
                torch.tensor([e[3][i] for e in mini_batch], device=device, dtype=float)
                for i in range(self.nber_ag)
            ],
        )

        return s, s_prime, actions, reward

    def train(self):
        cl, al = [], []
        for i, agent_i in enumerate(self.agent_list):

            agent_i.mu.train()

            s, s_p, a, r = self.sample()
            a_p = [agent.mu_target(s_p[i]) for i, agent in enumerate(self.agent_list)]

            # --critic--#
            y = r[i].unsqueeze(-1) + self.gamma * agent_i.q(s_p[i], a_p)
            q = agent_i.q(s[i], a)
            critic_loss = F.smooth_l1_loss(q, y.detach())

            # --actor--#
            mu = [agent.mu(s[i]) for i, agent in enumerate(self.agent_list)]
            agent_i.q.requires_grad_(False)
            actor_loss = agent_i.q(s[i], mu).mean(dim=0)

            # --optim--#
            agent_i.optim.zero_grad()
            (critic_loss - actor_loss).backward()
            agent_i.optim.step()
            agent_i.q.requires_grad_(True)

            # --add record--#
            cl.append(critic_loss.item())
            al.append(actor_loss.item())

        for agent in self.agent_list:
            soft_update(agent.q_target, agent.q, rho=self.rho)

        self.iteration += 1

        return cl, al


def soft_update(net, net_target, rho=0):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * rho + param.data * (1 - rho))


def phi(ob):
    """
	Return formatted state concatenation
	"""
    return [torch.tensor(s_ag, dtype=float, device=device) for s_ag in ob]
