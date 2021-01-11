import numpy as np
import torch


class Memory:
    """A Memory which stores transitions: `(start_state, action, end_state, reward, done)`.
    When the maximum memory size is reached, adding a new transition replaces a random old transition."""

    def __init__(self, max_size, state_size):
        self.max_size = max_size
        self.state_size = state_size

        self.length = 0

        self.start_states = torch.zeros((max_size, state_size), dtype=torch.float)
        self.actions = torch.zeros((max_size, 1), dtype=torch.float)
        self.end_states = torch.zeros((max_size, state_size), dtype=torch.float)
        self.rewards = torch.zeros((max_size, 1), dtype=torch.float)
        self.dones = torch.zeros((max_size, 1), dtype=torch.float)

    def __len__(self):
        return self.length

    @property
    def is_full(self):
        return len(self) == self.max_size

    def store(self, start_state, action, end_state, reward, done):
        """
        Store a transition in the memory.

        :param start_state: a torch Tensor of shape `(state_size,)`, dtype `torch.float`.
        :param action: a torch Tensor of shape `(1,)`, dtype `torch.float`.
        :param end_state: a torch Tensor of shape `(state_size,)`, dtype `torch.float`.
        :param reward: a torch Tensor of shape `(1,)`, dtype `torch.float`.
        :param done: a torch Tensor of shape `(1,)`, dtype `torch.float`.
        :return: the index of the inserted transition.
        """

        if (
            type(start_state) != torch.Tensor
            or start_state.shape != (self.state_size,)
            or start_state.dtype != torch.float
        ):
            raise ValueError(
                "`start_state` must be a `torch.Tensor` of shape `(state_size,)` and dtype `torch.float`."
            )

        if (
            type(action) != torch.Tensor
            or action.shape != (1,)
            or action.dtype != torch.float
        ):
            raise ValueError(
                "`action` must be a `torch.Tensor` of shape `(1,)` and dtype `torch.float`."
            )

        if (
            type(end_state) != torch.Tensor
            or end_state.shape != (self.state_size,)
            or end_state.dtype != torch.float
        ):
            raise ValueError(
                "`end_state` must be a `torch.Tensor` of shape `(state_size,)` and dtype `torch.float`."
            )

        if (
            type(reward) != torch.Tensor
            or reward.shape != (1,)
            or reward.dtype != torch.float
        ):
            raise ValueError(
                "`reward` must be a `torch.Tensor` of shape `(1,)` and dtype `torch.float`."
            )

        if (
            type(done) != torch.Tensor
            or done.shape != (1,)
            or done.dtype != torch.float
        ):
            raise ValueError(
                "`done` must be a `torch.Tensor` of shape `(1,)` and dtype `torch.float`."
            )

        # If the memory is full, replace a random transition.
        # Otherwise, put the transition at the end of the memory.
        index = np.random.randint(len(self)) if self.is_full else self.length

        self.start_states[index] = start_state
        self.actions[index] = action
        self.end_states[index] = end_state
        self.rewards[index] = reward
        self.dones[index] = done

        self.length = min(self.length + 1, self.max_size)

        return index

    def sample(self, n):
        """
        :return: a tuple `(start_state, action, end_state, reward, done)`. Each element of the tuple is a tensor.
        """

        indices = np.random.choice(len(self), n)

        return (
            self.start_states[indices],
            self.actions[indices],
            self.end_states[indices],
            self.rewards[indices],
            self.dones[indices],
        )


class PrioritizedMemory:
    def __init__(
        self, max_size, state_size, alpha=1, beta=1, epsilon=0.1, max_priority=1
    ):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.max_priority = max_priority

        self.memory = Memory(max_size, state_size)
        self.priorities = max_priority * np.ones((max_size,))

    def __len__(self):
        return len(self.memory)

    @property
    def is_full(self):
        return self.memory.is_full

    def store(self, start_state, action, end_state, reward, done):
        """
        Store a transition in the memory. This transition is given a priority `self.max_priority`. 

        :param start_state: a torch Tensor of shape `(state_size,)`, dtype `torch.float`.
        :param action: a torch Tensor of shape `(1,)`, dtype `torch.float`.
        :param end_state: a torch Tensor of shape `(state_size,)`, dtype `torch.float`.
        :param reward: a torch Tensor of shape `(1,)`, dtype `torch.float`.
        :param done: a torch Tensor of shape `(1,)`, dtype `torch.float`.
        :return: the index of the inserted transition.
        """

        index = self.memory.store(start_state, action, end_state, reward, done)
        self.priorities[index] = self.max_priority
        return index

    def update_priorities(self, indices, priorities):
        """Update the priority of the `indices` transitions.

        This priority is at most equal to `self.max_priority`.
        """

        self.priorities[indices] = np.minimum(self.max_priority, priorities)

    def sample(self, n):
        """
        :return: a tuple `(indices, start_state, action, end_state, reward, done, weight)`.
            Each element of the tuple is a tensor, except `indices` and `weights` which are `np.ndarray`.
        """

        probabilities = (self.priorities + self.epsilon) ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self), n, replace=False, p=probabilities)
        weights = (1 / len(self) * 1 / probabilities) ** self.beta

        return (
            indices,
            self.memory.start_states[indices],
            self.memory.actions[indices],
            self.memory.end_states[indices],
            self.memory.rewards[indices],
            self.memory.dones[indices],
            weights[indices],
        )
