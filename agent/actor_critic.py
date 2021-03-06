import torch
from torch import nn
import torch.nn.functional as F


class ActorCriticNetwork(nn.Module):
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


class ActorCritic:
    def __init__(self, observation_space, action_space, learning_rate, gamma):
        super().__init__()

        self.network = ActorCriticNetwork(observation_space, action_space)

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
        """Return the action obtained from `state`."""

        probabilities = self.network.policy(state)
        return torch.distributions.Categorical(probabilities).sample().item()

    def add_transition(self, transition):
        """Add a transition to the agent's memory.

        :param transition: `(start_state, action, reward, end_state, done)`.
        """

        self.transitions.append(transition)

    def optimize(self):
        """Optimize the agent's networks."""

        start_states, actions, rewards, end_states, dones = self._get_batch()

        # Value loss.
        td_0 = rewards + self.gamma * self.network.value(end_states) * dones
        value_loss = F.smooth_l1_loss(self.network.value(start_states), td_0.detach())

        # Policy objective.
        advantage = td_0 - self.network.value(start_states)
        probabilities = self.network.policy(start_states, softmax_dim=1)
        action_probabilities = probabilities.gather(1, actions)
        objective = -torch.log(action_probabilities) * advantage.detach()

        self.optimizer.zero_grad()
        value_loss.backward()
        objective.mean().backward()
        self.optimizer.step()
