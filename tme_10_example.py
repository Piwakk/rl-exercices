import matplotlib

matplotlib.use("TkAgg")
import numpy as np

from tme_10 import make_env


if __name__ == "__main__":
    env, scenario, world = make_env("simple_spread")

    o = env.reset()
    reward = []

    print(env.observation_space)
    print(o[0].shape)

    for _ in range(100):
        a = []
        for i, _ in enumerate(env.agents):
            a.append(np.random.rand(2))
        o, r, d, i = env.step(a)
        print(o, r, d, i)

        reward.append(r)
        env.render(mode="none")
    print(reward)

    env.close()
