from itertools import product

import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from agent import ActorCritic
from logger import get_logger
from experiment import Experiment


number_of_episodes = 2000
optimize_every = 10  # Number of steps.
show_every = 100  # Number of episodes.


if __name__ == "__main__":
    for learning_rate, gamma in product(
        (0.0001, 0.001, 0.01, 0.1, 1), (0.98, 0.99, 0.999)
    ):
        env = gym.make("LunarLander-v2")

        # Create a new agent here.
        experiment = Experiment.create(
            base_name="actor_critic/actor_critic_LunarLander-v2",
            model_class=ActorCritic,
            hp={
                "observation_space": env.observation_space,
                "action_space": env.action_space,
                "learning_rate": learning_rate,
                "gamma": gamma,
            },
        )
        experiment.save()

        # Or load a previous one.
        # experiment = Experiment.load("...")

        logger = get_logger(experiment.name, file_path=experiment.log_path)
        writer = SummaryWriter(
            log_dir=experiment.writer_path, purge_step=experiment.episode
        )
        experiment.info(logger)

        while experiment.episode < number_of_episodes:
            experiment.episode += 1
            show = (experiment.episode + 1) % show_every == 0

            state = env.reset()
            episode_reward, episode_steps = 0, 0

            while True:
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
                state = end_state
                experiment.step += 1
                episode_steps += 1
                episode_reward += reward

                # Optimize if needed.
                if (experiment.step + 1) % optimize_every == 0:
                    experiment.model.optimize()

                if done:
                    break

            # Log.
            if show:
                logger.info(f"Episode {experiment.episode}: reward = {episode_reward}.")

            writer.add_scalars(
                "train",
                {"reward": episode_reward, "steps": episode_steps},
                global_step=experiment.episode,
            )

        env.close()
