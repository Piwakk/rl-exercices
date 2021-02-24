import copy

import numpy as np
import torch
from torch import nn

from memory import Memory, PrioritizedMemory


class NN(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):
        super(NN, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))

    def setcuda(self, device):
        self.cuda(device=device)

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.tanh(x)
            x = self.layers[i](x)

        return x


class DQNAgent:
    def __init__(
        self,
        env,
        loss_function,
        memory_size,
        batch_size,
        epsilon_0,
        gamma,
        lr,
        q_layers,
        sync_frequency=None,
        device=None,
    ):
        """
        :param sync_frequency: if not `None`, the number of steps between each synchronization of the target network. Otherwise, no target network is used.
        """

        self.env = env
        self.action_space = env.action_space

        self.loss_function = loss_function
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_0 = epsilon_0
        self.gamma = gamma
        self.lr = lr
        self.q_layers = q_layers
        self.sync_frequency = sync_frequency

        self.memory = Memory(
            max_size=memory_size, state_size=env.observation_space.shape[0]
        )
        self.q = NN(env.observation_space.shape[0], env.action_space.n, q_layers)
        self.target_q = None if self.sync_frequency is None else copy.deepcopy(self.q)
        self.optim = torch.optim.SGD(params=self.q.parameters(), lr=lr)

        self.train_steps = 0
        self.train_steps_since_last_sync = 0
        self.last_state = None
        self.last_action = None

        if device is not None:
            self.q.to(device)

            if self.target_q is not None:
                self.target_q.to(device)

    def _get_state(self, observation):
        return torch.tensor(observation, dtype=torch.float)

    def step_test(self, observation):
        return int(torch.argmax(self.q.forward(self._get_state(observation))))

    def step_train(self, observation):
        """Return the action and the loss.
        The state corresponding to `observation` and the returned action are remembered
        (they might be used later by the method `self.record`)."""

        state = self._get_state(observation)

        # Optimize `q` if there are enough experiments in the buffer.
        loss = self._optimize_q() if self.memory.is_full else None

        # Choose the action.
        if np.random.rand(1) > 1 / (1 + self.epsilon_0 * self.train_steps):
            action = int(torch.argmax(self.q.forward(state)))  # Exploit.
        else:
            action = np.random.randint(self.action_space.n)  # Explore.

        # Update the iteration variables.
        self.train_steps += 1
        self.train_steps_since_last_sync += 1
        self.last_state = state
        self.last_action = action

        # Sync the target network if needed.
        if (
            self.sync_frequency is not None
            and self.train_steps_since_last_sync >= self.sync_frequency
        ):
            self.target_q = copy.deepcopy(self.q)
            self.train_steps_since_last_sync = 0

        return action, loss

    def _optimize_q(self):
        """Run a step in the gradient descent algorithm to optimize `q`."""

        start_states, actions, end_states, rewards, dones = self.memory.sample(
            n=self.batch_size
        )
        self.optim.zero_grad()

        # Compute the learning tensor.
        mask = torch.tensor(
            [
                [
                    1.0 if action == col_index else 0.0
                    for col_index in range(self.action_space.n)
                ]
                for row_index, action in enumerate(actions)
            ]
        )
        learning = torch.mul(mask, self.q.forward(start_states)).sum(dim=1)

        # Compute the target.
        with torch.no_grad():
            coefficients = (1 - dones) * self.gamma

            if self.target_q is None:
                target_q_values = self.q.forward(end_states).max(dim=1).values
            else:
                target_q_values = self.target_q.forward(end_states).max(dim=1).values

            target = rewards.squeeze() + torch.mul(
                coefficients.squeeze(), target_q_values
            )

        # Optimize.
        loss = self.loss_function(learning, target)
        loss.backward()
        self.optim.step()

        return loss

    def record(self, observation, reward, done):
        """Record the transition `(self.last_state, self.last_action, observation, reward, done)`."""

        if self.last_state is not None and self.last_action is not None:
            self.memory.store(
                start_state=self.last_state,
                action=torch.tensor([self.last_action], dtype=torch.float),
                end_state=torch.tensor(observation, dtype=torch.float),
                reward=torch.tensor([reward], dtype=torch.float),
                done=torch.tensor([done], dtype=torch.float),
            )


class DuelingDQN(torch.nn.Module):
    def __init__(
        self, observation_dim, number_of_actions, advantage_layers, value_layers
    ):
        super(DuelingDQN, self).__init__()

        self.observation_dim = observation_dim
        self.number_of_actions = number_of_actions

        self.advantage_function = NN(
            observation_dim, number_of_actions, advantage_layers
        )
        self.value_function = NN(observation_dim, 1, value_layers)

    def forward(self, observation):
        value = self.value_function(observation)
        advantage = self.advantage_function(observation)
        return (
            value
            + advantage
            - advantage.sum(dim=-1).reshape(-1, 1) / self.number_of_actions
        )


class DuelingDQNAgent:
    def __init__(
        self,
        env,
        loss_function,
        memory_size,
        batch_size,
        epsilon_0,
        gamma,
        lr,
        advantage_layers,
        value_layers,
        memory_alpha=1,
        memory_beta=1,
        sync_frequency=None,
    ):
        """
        :param sync_frequency: if not `None`, the number of steps between each synchronization of the target network. Otherwise, no target network is used.
        """

        self.env = env
        self.action_space = env.action_space

        self.loss_function = loss_function
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_0 = epsilon_0
        self.gamma = gamma
        self.lr = lr
        self.advantage_layers = advantage_layers
        self.value_layers = value_layers
        self.memory_alpha = memory_alpha
        self.memory_beta = memory_beta
        self.sync_frequency = sync_frequency

        self.memory = PrioritizedMemory(
            max_size=memory_size,
            state_size=env.observation_space.shape[0],
            alpha=memory_alpha,
            beta=memory_beta,
        )
        self.q = DuelingDQN(
            env.observation_space.shape[0],
            env.action_space.n,
            advantage_layers,
            value_layers,
        )
        self.target_q = None if self.sync_frequency is None else copy.deepcopy(self.q)
        self.optim = torch.optim.SGD(params=self.q.parameters(), lr=lr)

        self.train_steps = 0
        self.train_steps_since_last_sync = 0
        self.last_state = None
        self.last_action = None

    def _get_state(self, observation):
        return torch.tensor(observation, dtype=torch.float)

    def step_test(self, observation):
        return int(torch.argmax(self.q.forward(self._get_state(observation))))

    def step_train(self, observation):
        """Return the action and the loss.
        The state corresponding to `observation` and the returned action are remembered
        (they might be used later by the method `self.record`)."""

        state = self._get_state(observation)

        # Optimize `q` if there are enough experiments in the buffer.
        loss = self._optimize_q() if self.memory.is_full else None

        # Choose the action.
        if np.random.rand(1) > 1 / (1 + self.epsilon_0 * self.train_steps):
            action = int(torch.argmax(self.q.forward(state)))  # Exploit.
        else:
            action = np.random.randint(self.action_space.n)  # Explore.

        # Update the iteration variables.
        self.train_steps += 1
        self.train_steps_since_last_sync += 1
        self.last_state = state
        self.last_action = action

        # Sync the target network if needed.
        if (
            self.sync_frequency is not None
            and self.train_steps_since_last_sync >= self.sync_frequency
        ):
            self.target_q = copy.deepcopy(self.q)
            self.train_steps_since_last_sync = 0

        return action, loss

    def _optimize_q(self):
        """Run a step in the gradient descent algorithm to optimize `q`."""

        (
            indices,
            start_states,
            actions,
            end_states,
            rewards,
            dones,
            weights,
        ) = self.memory.sample(n=self.batch_size)
        self.optim.zero_grad()

        # Compute the learning tensor.
        mask = torch.tensor(
            [
                [
                    1.0 if action == col_index else 0.0
                    for col_index in range(self.action_space.n)
                ]
                for row_index, action in enumerate(actions)
            ]
        )
        learning = torch.mul(mask, self.q.forward(start_states)).sum(dim=1)

        # Compute the target.
        with torch.no_grad():
            coefficients = (1 - dones) * self.gamma

            if self.target_q is None:
                target_q_values = self.q.forward(end_states).max(dim=1).values
            else:
                target_q_values = self.target_q.forward(end_states).max(dim=1).values

            target = rewards.squeeze() + torch.mul(
                coefficients.squeeze(), target_q_values
            )

        # Update the priorities in the memory.
        with torch.no_grad():
            priorities = torch.abs(target - learning).detach().numpy()
            self.memory.update_priorities(indices, priorities)

        # Optimize.
        loss = torch.mul(
            self.loss_function(learning, target), torch.tensor(weights / weights.max())
        ).mean()
        loss.backward()
        self.optim.step()

        return loss

    def record(self, observation, reward, done):
        """Record the transition `(self.last_state, self.last_action, observation, reward, done)`."""

        if self.last_state is not None and self.last_action is not None:
            self.memory.store(
                start_state=self.last_state,
                action=torch.tensor([self.last_action], dtype=torch.float),
                end_state=torch.tensor(observation, dtype=torch.float),
                reward=torch.tensor([reward], dtype=torch.float),
                done=torch.tensor([done], dtype=torch.float),
            )


class PrioritizedDQNAgent:
    def __init__(
        self,
        env,
        loss_function,
        memory_size,
        batch_size,
        epsilon_0,
        gamma,
        lr,
        q_layers,
        memory_alpha=1,
        memory_beta=1,
        sync_frequency=None,
    ):
        """
        :param sync_frequency: if not `None`, the number of steps between each synchronization of the target network. Otherwise, no target network is used.
        """

        self.env = env
        self.action_space = env.action_space

        self.loss_function = loss_function
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_0 = epsilon_0
        self.gamma = gamma
        self.lr = lr
        self.q_layers = q_layers
        self.memory_alpha = memory_alpha
        self.memory_beta = memory_beta
        self.sync_frequency = sync_frequency

        self.memory = PrioritizedMemory(
            max_size=memory_size,
            state_size=env.observation_space.shape[0],
            alpha=memory_alpha,
            beta=memory_beta,
        )
        self.q = NN(env.observation_space.shape[0], env.action_space.n, q_layers)
        self.target_q = None if self.sync_frequency is None else copy.deepcopy(self.q)
        self.optim = torch.optim.SGD(params=self.q.parameters(), lr=lr)

        self.train_steps = 0
        self.train_steps_since_last_sync = 0
        self.last_state = None
        self.last_action = None

    def _get_state(self, observation):
        return torch.tensor(observation, dtype=torch.float)

    def step_test(self, observation):
        return int(torch.argmax(self.q.forward(self._get_state(observation))))

    def step_train(self, observation):
        """Return the action and the loss.
        The state corresponding to `observation` and the returned action are remembered
        (they might be used later by the method `self.record`)."""

        state = self._get_state(observation)

        # Optimize `q` if there are enough experiments in the buffer.
        loss = self._optimize_q() if self.memory.is_full else None

        # Choose the action.
        if np.random.rand(1) > 1 / (1 + self.epsilon_0 * self.train_steps):
            action = int(torch.argmax(self.q.forward(state)))  # Exploit.
        else:
            action = np.random.randint(self.action_space.n)  # Explore.

        # Update the iteration variables.
        self.train_steps += 1
        self.train_steps_since_last_sync += 1
        self.last_state = state
        self.last_action = action

        # Sync the target network if needed.
        if (
            self.sync_frequency is not None
            and self.train_steps_since_last_sync >= self.sync_frequency
        ):
            self.target_q = copy.deepcopy(self.q)
            self.train_steps_since_last_sync = 0

        return action, loss

    def _optimize_q(self):
        """Run a step in the gradient descent algorithm to optimize `q`."""

        (
            indices,
            start_states,
            actions,
            end_states,
            rewards,
            dones,
            weights,
        ) = self.memory.sample(n=self.batch_size)
        self.optim.zero_grad()

        # Compute the learning tensor.
        mask = torch.tensor(
            [
                [
                    1.0 if action == col_index else 0.0
                    for col_index in range(self.action_space.n)
                ]
                for row_index, action in enumerate(actions)
            ]
        )
        learning = torch.mul(mask, self.q.forward(start_states)).sum(dim=1)

        # Compute the target.
        with torch.no_grad():
            coefficients = (1 - dones) * self.gamma

            if self.target_q is None:
                target_q_values = self.q.forward(end_states).max(dim=1).values
            else:
                target_q_values = self.target_q.forward(end_states).max(dim=1).values

            target = rewards.squeeze() + torch.mul(
                coefficients.squeeze(), target_q_values
            )

        # Update the priorities in the memory.
        with torch.no_grad():
            priorities = torch.abs(target - learning).detach().numpy()
            self.memory.update_priorities(indices, priorities)

        # Optimize.
        loss = torch.mul(
            self.loss_function(learning, target), torch.tensor(weights / weights.max())
        ).mean()
        loss.backward()
        self.optim.step()

        return loss

    def record(self, observation, reward, done):
        """Record the transition `(self.last_state, self.last_action, observation, reward, done)`."""

        if self.last_state is not None and self.last_action is not None:
            self.memory.store(
                start_state=self.last_state,
                action=torch.tensor([self.last_action], dtype=torch.float),
                end_state=torch.tensor(observation, dtype=torch.float),
                reward=torch.tensor([reward], dtype=torch.float),
                done=torch.tensor([done], dtype=torch.float),
            )
