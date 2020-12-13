import torch
from torch import nn
import torch.nn.functional as F


class PPONetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        self.fc_1 = nn.Linear(observation_space.shape[0], 256)
        self.fc_policy = nn.Linear(256, action_space.n)
        self.fc_value = nn.Linear(256, 1)

    def policy(self, x, softmax_dim=0):
        x = self.fc_policy(F.relu(self.fc_1(x)))
        return F.softmax(x, dim=softmax_dim)

    def value(self, x):
        return self.fc_value(F.relu(self.fc_1(x)))


class PPOAdaptive:
    def __init__(self, observation_space, action_space, learning_rate, gamma, delta, k):
        super().__init__()

        self.transitions = []
        self.k = k
        self.beta = 1
        self.delta = delta
        self.network = PPONetwork(observation_space, action_space)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.transitions = []

    def _get_batch(self):
        """Transform `self.transitions` into a tuple of tensors then empty it.

        :return: the content of `self.transitions` as a tuple of tensors
            `(start_states, actions, rewards, end_states, dones)`.
        """

        start_states, actions, rewards, end_states, dones = [], [], [], [], []

        for transition in self.transitions:
            start_states.append(transition[0])
            actions.append([transition[1]])
            rewards.append([transition[2] / 100.0])
            end_states.append(transition[3])
            dones.append([0.0 if transition[4] else 1.0])

        self.transitions = []

        return (
            torch.tensor(start_states, dtype=torch.float),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float),
            torch.tensor(end_states, dtype=torch.float),
            torch.tensor(dones, dtype=torch.float),
        )

    def step(self, state):
        """Return the action obtained from `state` and the policy entropy."""

        probabilities = self.network.policy(state)
        entropy = -(probabilities * torch.log(probabilities)).sum()
        return (
            torch.distributions.Categorical(probabilities).sample().item(),
            entropy.item(),
        )

    def add_transition(self, transition):
        """Add a transition to the agent's memory.

        :param transition: `(start_state, action, reward, end_state, done)`.
        """

        self.transitions.append(transition)

    def _optimize_value(self, start_states, actions, rewards, end_states, dones):
        td_0 = rewards + self.gamma * self.network.value(end_states) * dones
        loss = F.smooth_l1_loss(self.network.value(start_states), td_0.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _optimize_policy(self, start_states, actions, rewards, end_states, dones):
        # Compute the old advantage.
        td_0 = rewards + self.gamma * self.network.value(end_states) * dones
        old_advantage = (td_0 - self.network.value(start_states)).detach()
        old_probabilities = self.network.policy(start_states, softmax_dim=1).detach()
        old_action_probabilities = old_probabilities.gather(1, actions).detach()

        losses = torch.zeros(self.k)

        # Optimize the policy.
        for i in range(self.k):
            new_probabilities = self.network.policy(start_states, softmax_dim=1)
            new_action_probabilities = new_probabilities.gather(1, actions)

            # aka $\mathcal{L}_{\theta_k}$.
            l = (
                old_advantage * new_action_probabilities / old_action_probabilities
            ).mean()

            # aka $D_{KL(\theta_k | \theta)}$.
            d_kl = torch.distributions.kl_divergence(
                torch.distributions.Categorical(old_probabilities),
                torch.distributions.Categorical(new_probabilities),
            ).mean()

            loss = -(l - self.beta * d_kl)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses[i] = loss.item()

        # Update `self.beta`.
        new_probabilities = self.network.policy(start_states, softmax_dim=1)
        d_kl = torch.distributions.kl_divergence(
            torch.distributions.Categorical(old_probabilities),
            torch.distributions.Categorical(new_probabilities),
        ).mean()

        if d_kl >= 1.5 * self.delta:
            self.beta *= 2
        elif d_kl <= self.delta / 1.5:
            self.beta /= 2

        return losses.mean().item(), d_kl.item()

    def optimize(self):
        """Optimize the agent's networks."""

        start_states, actions, rewards, end_states, dones = self._get_batch()
        policy_loss, d_kl = self._optimize_policy(
            start_states, actions, rewards, end_states, dones
        )
        value_loss = self._optimize_value(
            start_states, actions, rewards, end_states, dones
        )

        return policy_loss, d_kl, value_loss
