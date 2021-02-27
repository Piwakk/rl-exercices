from itertools import product

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agent import DDPG
from experiment import Experiment
from logger import get_logger


number_of_episodes = 500
show_every = 10  # Number of episodes.
optimize_every = 1  # Number of steps.
test_after = 300  # Number of episodes.


if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0")

    # Create a new agent here.
    experiment = Experiment.create(
        base_name="ddpg/ddpg_MountainCarContinuous-v0",
        model_class=DDPG,
        hp={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
            "number_of_updates": 3,
            "policy_learning_rate": 1e-2,
            "q_learning_rate": 5e-3,
            "noise_sigma": 0.5,
            "memory_max_size": 10000,
            "batch_size": 256,
            "gamma": 0.99,
            "rho": 0.999,
        },
    )

    # Or load a previous one.
    # experiment = Experiment.load("ddpg/ddpg_Pendulum-v0__20201214_2050")

    logger = get_logger(experiment.name, file_path=experiment.log_path)
    writer = SummaryWriter(
        log_dir=experiment.writer_path, purge_step=experiment.episode
    )
    experiment.info(logger)

    last_episode_rewards = []

    while experiment.episode < number_of_episodes:
        experiment.episode += 1
        show = (experiment.episode + 1) % show_every == 0
        is_train = experiment.episode <= test_after

        state = env.reset()
        episode_reward, episode_steps = 0, 0
        policy_losses, q_losses = [], []

        while True:
            if is_train:
                # Draw an action and act on the environment.
                action = experiment.model.step(torch.from_numpy(state).float())
                end_state, reward, done, info = env.step(action)

                # Record the transition.
                experiment.model.add_transition(
                    (
                        state,
                        action,
                        reward,
                        end_state,
                        False if info.get("TimeLimit.truncated") else done,
                    )
                )

                # Optimize if needed.
                if (experiment.step + 1) % optimize_every == 0:
                    q_loss, policy_loss = experiment.model.optimize()
                    policy_losses.append(policy_loss)
                    q_losses.append(q_loss)
            else:
                # Draw an action and act on the environment.
                action = experiment.model.step(
                    torch.from_numpy(state).float(),
                    train=False,
                    episode=experiment.episode,
                )
                end_state, reward, done, info = env.step(action)

            state = end_state
            experiment.step += 1
            episode_steps += 1
            episode_reward += reward

            if show:
                env.render()

            if done:
                break

        last_episode_rewards.append(episode_reward)
        experiment.save()

        # Log.
        if show:
            logger.info(
                f"Episode {experiment.episode} ({'train' if is_train else 'test'})"
            )
            logger.info(f"\tlast_rewards = {sum(last_episode_rewards) / show_every}.")
            last_episode_rewards = []

        if is_train:
            writer.add_scalars(
                "train",
                {"reward": episode_reward, "steps": episode_steps},
                global_step=experiment.episode,
            )
        else:
            writer.add_scalars(
                "test",
                {"reward": episode_reward, "steps": episode_steps},
                global_step=experiment.episode,
            )

        if len(policy_losses) > 0:
            writer.add_scalars(
                "debug",
                {"q_loss": np.mean(q_losses), "policy_loss": np.mean(policy_losses)},
                global_step=experiment.episode,
            )

    env.close()
