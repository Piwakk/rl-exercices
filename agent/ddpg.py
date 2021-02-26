import copy

import torch
from torch import nn
import torch.nn.functional as F

from memory import Memory


class DDPG:
    def __init__(
        self,
        observation_space,
        action_space,
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
            nn.Linear(observation_space.shape[0], 20),
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

        self.q = nn.Sequential(
            nn.Linear(observation_space.shape[0] + action_space.shape[0], 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=q_learning_rate)
        self.q_target = copy.deepcopy(self.q)

        self.action_space = action_space
        self.action_low = action_space.low[0]
        self.action_high = action_space.high[0]
        self.noise_sigma = noise_sigma

        self.memory = Memory(
            memory_max_size,
            observation_space.shape[0],
            action_size=action_space.shape[0],
        )
        self.batch_size = batch_size

        self.gamma = gamma
        self.rho = rho

    def step(self, state, train=True):
        """Return the action obtained from `state`.

        :param train: if `True`, add a random noise."""

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
            .view(-1)
            .detach()
            .numpy()
        )

    def add_transition(self, transition):
        """Add a transition to the agent's memory.

        :param transition: `(start_state, action, reward, end_state, done)`.
        """

        (start_state, action, reward, end_state, done) = transition

        return self.memory.store(
            torch.tensor(start_state, dtype=torch.float),
            torch.tensor(action, dtype=torch.float),
            torch.tensor(end_state, dtype=torch.float),
            torch.tensor([reward], dtype=torch.float),
            torch.tensor([1.0 if done else 0.0], dtype=torch.float),
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
