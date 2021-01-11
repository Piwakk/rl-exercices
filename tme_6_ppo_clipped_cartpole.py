import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agent import PPOAdaptive, PPOClipped
from experiment import Experiment
from logger import get_logger


number_of_episodes = 3000
optimize_every = 64  # Number of steps.
show_every = 100  # Number of episodes.


if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    # Create a new agent here.
    experiment = Experiment.create(
        base_name="ppo_clipped_CartPole-v1",
        model_class=PPOClipped,
        hp={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
            "learning_rate": 0.0005,
            "gamma": 0.99,
            "k": 3,
            "epsilon": 1e-1,
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
        entropies, policy_losses, value_losses = [], [], []

        while True:
            # Draw an action and act on the environment.
            action, entropy = experiment.model.step(torch.from_numpy(state).float())
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
            entropies.append(entropy)

            # Optimize if needed.
            if (experiment.step + 1) % optimize_every == 0:
                policy_loss, value_loss = experiment.model.optimize()
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)

            # Show if needed.
            if show:
                env.render()

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

        debug_scalars = {"entropy": np.mean(entropies)}

        if len(policy_losses) > 0:
            debug_scalars["policy_loss"] = np.mean(policy_losses)
            debug_scalars["value_loss"] = np.mean(value_losses)

        writer.add_scalars("debug", debug_scalars, global_step=experiment.episode)

    env.close()
