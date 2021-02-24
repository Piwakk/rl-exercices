import matplotlib

matplotlib.use("TkAgg")
import gym
import multiagent
import multiagent.scenarios
import multiagent.scenarios.simple_tag as simple_tag
import multiagent.scenarios.simple_tag as simple_spread
import multiagent.scenarios.simple_tag as simple_adversary
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from gym import wrappers, logger
import numpy as np
import copy
from utils import State, load_yaml, device, make_env, write_it, write_epoch
from ddpgMultiAgent import DDPGMultiAgent
from torch.utils.tensorboard import SummaryWriter


def __main__():

    # ---config---#
    config = load_yaml(f"./configs/maddpg.yaml")

    # ---env---#
    env, scenario, world = make_env(config.env)
    ob = env.reset()
    dim_state = len(ob[0])
    nber_ag = len(env.agents)
    print(f"{nber_ag} agents. Dimension of states: {dim_state}")

    # ---agents---#
    agent = DDPGMultiAgent(config)

    # ---writer---#
    run_id = f"env{config.env}_lr{config.lr}_gamma{config.gamma}_h_{config.h}_buffer{config.buffer_limit}_std{config.std}\
				_batch{config.mini_batch_size}_rho{config.rho}_ep{config.nber_ep}_act{config.nber_action}"
    writer = SummaryWriter(f"runs/{run_id}")

    for ep in range(config.nber_ep):

        reward = []
        verbose = agent.epoch % config.freq_verbose == 0

        for _ in range(config.nber_action):
            a = agent.act(ob)  # determine action
            ob_p, r, d, i = env.step(a)  # play action
            agent.store(ob, a, ob_p, r, d)  # store in buffer the transition
            reward.append(r)  # keep in record reward
            ob = ob_p  # update state

            # --train--#
            if (len(agent.memory) > config.mini_batch_size) and (
                agent.actions % config.freq_optim == 0
            ):
                cl, al = agent.train()
                write_it(writer, cl, al, r, agent.iteration)

            # --visualize--#
            if verbose:
                env.render(mode="none")

        mean_reward = np.array(reward).mean(axis=0)
        ob = env.reset()
        agent.epoch += 1
        write_epoch(writer, mean_reward, agent.epoch)
        print(f"Episode: {ep} - Reward: {mean_reward}")
    env.close()


if __name__ == "__main__":
    __main__()
