import numpy as np


class Q:
    """A basic implementation of the Q function for discrete states and actions."""

    def __init__(
        self,
        action_space,
        state_hasher=None,
        state_unhasher=None,
        default=0,
        action_to_str=None,
    ):
        self.action_space = action_space
        self.default = default
        self.state_hasher = state_hasher
        self.state_unhasher = state_unhasher
        self.action_to_str = action_to_str

        self._q = {}

    def _split_key(self, key):
        try:
            return key[0], key[1]
        except (TypeError, IndexError):
            raise IndexError(
                "A `Q` object should be indexed with a tuple (state, action)."
            )

    def _hash_state(self, state):
        return self.state_hasher(state) if self.state_hasher is not None else state

    def _unhash_state(self, state):
        return self.state_unhasher(state) if self.state_unhasher is not None else state

    def _str_action(self, action):
        if self.action_to_str is None:
            return str(action)

        return self.action_to_str.get(action, str(action))

    def __repr__(self):
        return self._q.__repr__()

    def __getitem__(self, key):
        state, action = self._split_key(key)
        state = self._hash_state(state)

        return self._q.get(state, {}).get(action, self.default)

    def __setitem__(self, key, value):
        state, action = self._split_key(key)
        state = self._hash_state(state)

        if action not in self.action_space:
            raise ValueError(f"`{action}` does not belong to the action space.")

        # A new state is created.
        if state not in self._q:
            self._q[state] = {_a: self.default for _a in self.action_space}

        self._q[state][action] = value

    def get_state(self, state):
        return self._q.get(
            self._hash_state(state), {_a: self.default for _a in self.action_space}
        )

    def items(self):
        return self._q.items()

    def values(self):
        return self._q.values()

    def keys(self):
        return self._q.keys()

    def max_for_state(self, state):
        """:return: a tuple `(action, q)` where `q` is the max for the given `state`.
        If `state` is not in the function, a random action and `self.default` are returned."""

        state = self._hash_state(state)

        if state not in self._q:
            return np.random.choice(self.action_space), self.default

        return max(list(self._q[state].items()), key=lambda x: x[1])

    def show(self):
        for state, values in self.items():
            print(self._unhash_state(state))
            print({self._str_action(action): values[action] for action in values})
            print(f"Best action: {self._str_action(self.max_for_state(state)[0])}\n")


class ActionStrategy:
    def choose(self, agent, state):
        raise NotImplementedError()


class EpsilonGreedy(ActionStrategy):
    """Explore if a random number between 0 and 1 is lower than `epsilon`, exploit otherwise."""

    def __init__(self, epsilon):
        self._epsilon = epsilon

    def choose(self, agent, state):
        if np.random.rand() >= self._epsilon:  # Exploit.
            return agent.q.max_for_state(state)[0]

        return np.random.choice(agent.action_space)  # Explore.


class EpsilonGreedyDecay(ActionStrategy):
    """Explore if a random number between 0 and 1 is lower than `epsilon`, exploit otherwise.
    `epsilon` is subject to a decay `alpha / global_t`"""

    def __init__(self, epsilon, alpha):
        self._epsilon = epsilon
        self._alpha = alpha

    def epsilon(self, global_t):
        return self._epsilon / ((global_t // self._alpha) + 1)

    def choose(self, agent, state):
        if np.random.rand() >= self.epsilon(agent.global_t):  # Exploit.
            return agent.q.max_for_state(state)[0]

        return np.random.choice(agent.action_space)  # Explore.


class Boltzmann(ActionStrategy):
    """Use the Boltzmann strategy with temperature `tau`."""

    def __init__(self, tau):
        self._tau = tau

    def choose(self, agent, state):
        exp_q_values = np.array(
            [np.exp(value / self._tau) for value in agent.q.get_state(state).values()]
        )
        exp_sum = np.sum(exp_q_values)
        probabilities = exp_q_values / exp_sum

        return np.random.choice(agent.action_space, p=probabilities)


state_hasher = str


def state_unhasher(s: str):
    """Return the Numpy array representation of a str state."""

    return np.array(
        [[int(x) for x in line[1:-1].split(" ")] for line in s[1:-1].split("\n ")]
    )


class UpdateStrategy:
    def update_q(self, agent, state, action, reward, next_state):
        raise NotImplementedError()


class OfflinePolicy(UpdateStrategy):
    """The strategy used by the regular Q Learning algorithm."""

    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma

    def update_q(self, agent, state, action, reward, next_state):
        agent.q[state, action] = (
            (1 - self.alpha) * agent.q[state, action]
            + self.alpha * (reward + self.gamma * agent.q.max_for_state(next_state)[1])
            # This is Q(next_state, maximising_action).
        )


class SARSA(UpdateStrategy):
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma

    def update_q(self, agent, state, action, reward, next_state):
        next_action = agent.action_strategy.choose(
            agent, next_state
        )  # Choose the next action.

        agent.q[state, action] = (1 - self.alpha) * agent.q[
            state, action
        ] + self.alpha * (reward + self.gamma * agent.q[next_state, next_action])


class DynaQ(UpdateStrategy):
    def __init__(self, alpha_q, gamma, alpha_mdp, k):
        """
        :param alpha_q: the parameter used to update the Q function.
        :param alpha_mdp: the parameter used to update the MDP (P and R functions).
        :param k: the number of samples drawn at each update.
        """

        self.alpha_q = alpha_q
        self.gamma = gamma
        self.alpha_mdp = alpha_mdp
        self.k = k

        self.rewards = {}  # The rewards component of the MDP.
        self.probabilities = {}  # The probabilities component of the MDP.

    def update_q(self, agent, state, action, reward, next_state):
        # Classic update of Q.
        agent.q[state, action] = (
            (1 - self.alpha_q) * agent.q[state, action]
            + self.alpha_q
            * (reward + self.gamma * agent.q.max_for_state(next_state)[1])
            # This is Q(next_state, maximising_action).
        )

        # Update the MDP.
        self.rewards[str(state), action, str(next_state)] = self.rewards.get(
            (str(state), action, str(next_state)), 0
        ) + self.alpha_mdp * (
            reward - self.rewards.get((str(state), action, str(next_state)), 0)
        )

        for s in agent.q.keys():
            if s == next_state:
                self.probabilities[str(state), action, s] = self.probabilities.get(
                    (str(state), action, s), 0
                ) + self.alpha_mdp * (
                    1 - self.probabilities.get((str(state), action, s), 0)
                )
            else:
                self.probabilities[str(state), action, s] = self.probabilities.get(
                    (str(state), action, s), 0
                ) + self.alpha_mdp * (
                    0 - self.probabilities.get((str(state), action, s), 0)
                )

        self._update_q_with_mdp(agent)

    def _update_q_with_mdp(self, agent):
        samples = zip(
            np.random.choice(list(agent.q.keys()), self.k),
            np.random.choice(agent.action_space, self.k),
        )

        for state, action in samples:
            agent.q[state, action] = (1 - self.alpha_q) * agent.q[
                state, action
            ] + self.alpha_q * sum(
                self.probabilities.get((state, action, next_state), 0)
                * (
                    self.rewards.get((state, action, next_state), 0)
                    + self.gamma * agent.q.max_for_state(next_state)[1]
                )
                for next_state in agent.q.keys()
            )


class QLearning:
    def __init__(
        self,
        env,
        action_space,
        observation_shape,
        action_strategy,
        update_strategy,
        action_to_str=None,
    ):
        self.env = env
        self.action_space = action_space
        self.observation_shape = observation_shape
        self.action_strategy = action_strategy
        self.update_strategy = update_strategy

        self.q = Q(
            action_space,
            state_hasher=str,
            state_unhasher=state_unhasher,
            action_to_str=action_to_str,
        )  # q[state, action]

        self.state = env.reset().copy()  # The current state.
        self.t = 0  # The time step for one run.
        self.global_t = 0  # The time step for the whole model.

    def reset(self):
        """Reset the environment and `self.t`. Do not reset `self.q` or `self.global_t`."""

        self.env.close()
        self.state = self.env.reset().copy()
        self.t = 0

    def _update_state(self, next_state):
        self.state = next_state.copy()
        self.t += 1
        self.global_t += 1

    def step(self):
        action = self.action_strategy.choose(self, self.state)

        next_state, reward, done, info = self.env.step(action)

        self.update_strategy.update_q(self, self.state, action, reward, next_state)
        self._update_state(next_state)

        return reward, done, info
