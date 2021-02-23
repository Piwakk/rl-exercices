import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from memory import Memory


class MADDPGSingleAgent:
    def __init__(
        self,
        observation_size,
        action_space,
        all_actions_size,
        number_of_updates,
        policy_learning_rate,
        q_learning_rate,
        noise_sigma,
        memory_max_size,
        batch_size,
        gamma,
        rho,
    ):
        self.number_of_updates = number_of_updates

        self.policy = nn.Sequential(
            nn.Linear(observation_size, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, action_space.shape[0]),
            nn.Tanh(),
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=policy_learning_rate
        )
        self.policy_target = copy.deepcopy(self.policy)

        q_in_features = observation_size + all_actions_size
        self.q = nn.Sequential(
            nn.Linear(q_in_features, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=q_learning_rate)
        self.q_target = copy.deepcopy(self.q)

        self.action_space = action_space
        self.all_actions_size = all_actions_size
        self.action_low = action_space.low[0]
        self.action_high = action_space.high[0]
        self.noise_sigma = noise_sigma

        self.memory = Memory(
            memory_max_size, observation_size, action_size=all_actions_size
        )
        self.batch_size = batch_size

        self.gamma = gamma
        self.rho = rho

    def add_transition(
        self, start_state, action, other_actions, end_state, reward, done
    ):
        """Add a transition to the agent's memory.

        By convention, `action` and `other_actions` are concatenated in that order and added to the memory."""

        return self.memory.store(
            torch.tensor(start_state, dtype=torch.float),
            torch.tensor(torch.hstack((action, other_actions)), dtype=torch.float),
            torch.tensor(end_state, dtype=torch.float),
            torch.tensor([reward], dtype=torch.float),
            torch.tensor([1.0 if done else 0.0], dtype=torch.float),
        )

    """
    def step(self, state, train=True):
        Return the action obtained from `state`.

        :param train: if `True`, add a random noise.

        # Map the output of `self.policy` to ]low, high[.
        action = (
            self.policy(state) * (self.action_high - self.action_low) / 2
            + (self.action_high + self.action_low) / 2
        )

        # Add some noise.
        if train:
            action += torch.randn(1) * self.noise_sigma

        return (
            torch.clip(action, self.action_low, self.action_high)
            .view(1)
            .detach()
            .numpy()
        )

    def _optimize_q(self, start_states, actions, end_states, rewards, dones):
        with torch.no_grad():
            targets = rewards + self.gamma * (1 - dones) * self.q_target(
                torch.hstack([end_states, self.policy_target(end_states)])
            )

        estimates = self.q(torch.hstack([start_states, actions]))
        loss = F.smooth_l1_loss(estimates, targets)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        return loss.item()

    def _optimize_policy(self, start_states, actions, end_states, rewards, dones):
        loss = -self.q(torch.hstack([start_states, self.policy(start_states)])).mean()

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        return loss.item()

    def _update_target(self):
        for target, source in zip(
            self.policy_target.parameters(), self.policy.parameters()
        ):
            target.data.copy_(target.data * self.rho + source.data * (1 - self.rho))

        for target, source in zip(self.q_target.parameters(), self.q.parameters()):
            target.data.copy_(target.data * self.rho + source.data * (1 - self.rho))

    def optimize(self):
        q_losses = torch.zeros(self.number_of_updates)
        policy_losses = torch.zeros(self.number_of_updates)

        for i in range(self.number_of_updates):
            sample = self.memory.sample(self.batch_size)
            q_losses[i] = self._optimize_q(*sample)
            policy_losses[i] = self._optimize_policy(*sample)
            self._update_target()

        return q_losses.mean().item(), policy_losses.mean().item()
    """


class MADDPG:
    def __init__(
        self,
        observation_spaces,
        action_spaces,
        number_of_updates,
        policy_learning_rate,
        q_learning_rate,
        noise_sigma,
        memory_max_size,
        batch_size,
        gamma,
        rho,
    ):
        """
        Multi-agent DDPG. The number of agents is inferred from `observation_space` and `action_space`.

        :param observation_spaces: list of the observation spaces.
        :param action_spaces:  list of the action spaces.
        :param number_of_updates: used for all the agents.
        :param policy_learning_rate: used for all the agents.
        :param q_learning_rate: used for all the agents.
        :param noise_sigma: used for all the agents.
        :param memory_max_size: used for all the agents.
        :param batch_size: used for all the agents.
        :param gamma: used for all the agents.
        :param rho: used for all the agents.
        """

        if len(observation_spaces) != len(action_spaces):
            raise ValueError(
                "`observation_spaces` and `action_spaces` have different lengths."
            )

        all_actions_size = sum(action_space.shape[0] for action_space in action_spaces)

        self.agents = [
            MADDPGSingleAgent(
                observation_space.shape[0],
                action_space,
                all_actions_size,
                number_of_updates,
                policy_learning_rate,
                q_learning_rate,
                noise_sigma,
                memory_max_size,
                batch_size,
                gamma,
                rho,
            )
            for observation_space, action_space in zip(
                observation_spaces, action_spaces
            )
        ]

    def add_transition(self, start_states, actions, rewards, end_states, dones):
        """Add a transition to the agents memories."""

        for i, agent in enumerate(self.agents):
            # Il faut split actions en actions[i] / actions[tout_sans_i]
            agent.add_transition(
                start_states[i],
                actions[i],
                actions[:i] + actions[i + 1 :],
                end_states[i],
                rewards[i],
                dones[i],
            )

    def step(self, states, train=True):
        """Return the action obtained from `states`.

        :param states:
        :param train: if `True`, add a random noise."""

        return [
            agent.step(torch.from_numpy(state).float(), train)
            for agent, state in zip(self.agents, states)
        ]

    def optimize(self):
        q_losses, policy_losses = [], []

        for agent in self.agents:
            q_loss, policy_loss = agent.optimize()
            q_losses.append(q_loss)
            policy_losses.append(policy_loss)

        return np.mean(q_losses), np.mean(policy_losses)
