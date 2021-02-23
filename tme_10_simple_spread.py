from gym.spaces import Box
import numpy as np

from agent.maddpg import MADDPG
from tme_10 import make_env


show_every = 10  # Number of episodes.

if __name__ == "__main__":
    env, scenario, world = make_env("simple_spread")
    episode = 0
    number_of_steps = 100
    number_of_episodes = 1000

    model = MADDPG(
        [Box(low=-np.inf, high=np.inf, shape=(14,)) for _ in env.observation_space],
        [Box(low=-1, high=1, shape=(2,)) for _ in env.action_space],
        number_of_updates=3,
        policy_learning_rate=5e-3,
        q_learning_rate=1e-3,
        noise_sigma=0.1,
        memory_max_size=10000,
        batch_size=1024,
        gamma=0.98,
        rho=0.995,
    )

    while episode < number_of_episodes:
        show = (episode + 1) % show_every == 0

        episode += 1
        episode_reward = 0
        states = env.reset()

        for step in range(number_of_steps):
            # actions = model.step(states)
            actions = [np.random.randn(2) for _ in states]
            end_states, rewards, dones, infos = env.step(actions)

            model.add_transition(
                states, actions, end_states, rewards, [False] * len(states)
            )

            episode_reward += sum(rewards)

            if show:
                env.render(mode="none")

        print(f"Episode reward = {episode_reward}")

    env.close()
