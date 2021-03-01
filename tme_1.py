import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")

raw_context = []
raw_click_rate = []

with open("tme_1.txt") as file:
    for line in file.readlines():
        raw_context.append(np.array(line.split(":")[1].split(";")).astype("float"))
        raw_click_rate.append(np.array(line.split(":")[2].split(";")).astype("float"))

raw_context = np.array(raw_context)
raw_click_rate = np.array(raw_click_rate)


class Strategy:
    def __init__(self, context: np.array, click_rate: np.array):
        self._context = context  # context[document_id, context_id]
        self._click_rate = click_rate  # click_rate[document_id, ad_id]
        self._actions = []  # actions[document_id]

        self.number_of_articles = self._click_rate.shape[0]
        self.number_of_ads = self._click_rate.shape[1]
        self.context_dimension = self._context.shape[1]

    @property
    def actions(self):
        return np.array(self._actions)

    @property
    def regret(self):
        """:return: the regret for each action."""

        if len(self._actions) == 0:
            raise RuntimeError("No action has been computed.")

        # Select the action with the best mean.
        static_best = self._click_rate[:, np.argmax(np.mean(self._click_rate, axis=0))]

        action_click_rate = self._click_rate[
            [range(self.number_of_articles)], self.actions.astype("int")
        ]

        return static_best - action_click_rate

    @property
    def cumulative_regret(self):
        return self.regret.cumsum()

    @property
    def cumulative_average_regret(self):
        return self.cumulative_regret / range(1, len(self.cumulative_regret) + 1)

    @property
    def average_regret(self):
        """:return: the average regret."""

        return np.sum(self.regret) / len(self._actions)

    def compute_action(self):
        """Should return an `actions` vector and save it to `self._actions`."""
        raise NotImplementedError()

    def plot_cumulative_average_regret(self, start=0, **kwargs):
        sns.lineplot(
            x=range(len(self.cumulative_average_regret[start:])),
            y=self.cumulative_average_regret[start:],
            **kwargs,
        )


class LoadedStrategy(Strategy):
    """Convenience class: `Strategy` loaded with the data."""

    def __init__(self):
        super().__init__(raw_context, raw_click_rate)


class RandomBaseline(LoadedStrategy):
    """Select a random action for each document."""

    def compute_action(self):
        self._actions = np.random.randint(10, size=self.number_of_articles)
        return self._actions


class OptimalBaseline(LoadedStrategy):
    """Select the best action for each document."""

    def compute_action(self):
        self._actions = np.argmax(self._click_rate, axis=1)
        return self._actions


class StaticBestBaseline(LoadedStrategy):
    """Select the action with the best _total_ average."""

    def compute_action(self):
        self._actions = np.argmax(np.mean(self._click_rate, axis=0)) * np.ones(
            self.number_of_articles
        )
        return self._actions


class IterativeBestBaseline(LoadedStrategy):
    """Select the action with the best average click rate at each iteration."""

    def compute_action(self):
        self._actions = np.array(
            [
                np.argmax(np.mean(self._click_rate[:t], axis=0))
                for t in range(0, self.number_of_articles)
            ]
        )
        return self._actions


class CountBasedStrategy(LoadedStrategy):
    def __init__(self):
        super().__init__()

        self._count = self.number_of_ads * [
            0
        ]  # The number of times each ad has been chosen so far.
        self._rewards = self.number_of_ads * [
            0
        ]  # The _average_ reward for each ad so far.

    def _add_action(self, ad_id: int):
        document_id = len(self._actions)
        old_count = self._count[ad_id]

        self._actions.append(ad_id)

        # Update `rewards`.
        action_reward = self._click_rate[document_id, ad_id]
        self._rewards[ad_id] = (old_count * self._rewards[ad_id] + action_reward) / (
            old_count + 1
        )

        self._count[ad_id] += 1

    def _initialize(self):
        """Explore each action."""

        for ad_id in range(self.number_of_ads):
            self._add_action(ad_id)


class EpsilonGreedy(CountBasedStrategy):
    def __init__(self, epsilon: float):
        super().__init__()
        self._epsilon = epsilon

    def compute_action(self):
        # Explore each action first.
        self._initialize()

        # Exploit and explore.
        for _ in range(self.number_of_ads, self.number_of_articles):
            if np.random.rand() > self._epsilon:
                self._add_action(
                    np.argmax(self._rewards)
                )  # Use the action with the highest reward.
            else:
                self._add_action(
                    np.random.randint(self.number_of_ads)
                )  # Use a random action.

        return self.actions


class UCB(CountBasedStrategy):
    def compute_action(self):
        # Explore each action first.
        self._initialize()

        # Exploit and explore.
        for t in range(self.number_of_ads, self.number_of_articles):
            # Compute the Bt for each action.
            bt = self.number_of_ads * [0]

            for ad_id in range(self.number_of_ads):
                mu = self._rewards[ad_id]  # The experimental mean so far.
                s = self._count[ad_id]  # The number of times this action was chosen.

                bt[ad_id] = self._rewards[ad_id] + np.sqrt(2 * np.log(t) / s)

            # Select the action_id.
            self._add_action(np.argmax(bt))

        return self.actions


class LinUCB(LoadedStrategy):
    def __init__(self, delta):
        super().__init__()

        self._alpha = 1 + np.sqrt(np.log(2 / delta) / 2)

        self._a = [
            np.identity(self.context_dimension) for _ in range(self.number_of_ads)
        ]

        self._b = [
            np.zeros((self.context_dimension, 1)) for _ in range(self.number_of_ads)
        ]

    def _compute_for_next_article(self, article_id):
        # article_id = t

        context = self._context[article_id].reshape((-1, 1))
        p = []  # p[ad_id] (or p[arm_id])

        for ad_id in range(self.number_of_ads):
            a = self._a[ad_id]
            inv_a = np.linalg.inv(a)
            b = self._b[ad_id]

            theta = np.matmul(inv_a, b)

            p.append(
                np.matmul(theta.T, context)
                + self._alpha * np.sqrt(np.matmul(np.matmul(context.T, inv_a), context))
            )

        next_action = np.argmax(p)
        reward = self._click_rate[article_id, next_action]

        self._a[next_action] = self._a[next_action] + np.matmul(context, context.T)
        self._b[next_action] = self._b[next_action] + reward * context

        self._actions.append(next_action)

    def compute_action(self):
        for article_id in range(self.number_of_articles):
            self._compute_for_next_article(article_id)

        return self.actions


if __name__ == "__main__":
    strategies = {
        "Optimal baseline": OptimalBaseline(),
        "Static best baseline": StaticBestBaseline(),
        # "Iterative best baseline": IterativeBestBaseline(),
        # "Random baseline": RandomBaseline(),
        "Epsilon greedy 0,01": EpsilonGreedy(epsilon=0.01),
        "Epsilon greedy 0,05": EpsilonGreedy(epsilon=0.05),
        "Upper-confidence bound": UCB(),
        "Lin-UCB 0,5": LinUCB(delta=0.5),
        "Lin-UCB 0,1": LinUCB(delta=0.1),
    }

    plt.figure(figsize=(20, 10))

    for strategy_name, strategy in strategies.items():
        strategy.compute_action()
        strategy.plot_cumulative_average_regret(start=10, label=strategy_name)

    plt.legend()
    plt.yticks(
        np.arange(-0.05, 0.15, 0.025),
        [round(x, 3) for x in np.arange(-0.05, 0.15, 0.025)],
    )
    plt.show()

    for strategy_name, strategy in strategies.items():
        print(f"{strategy_name}: {strategy.average_regret}")
