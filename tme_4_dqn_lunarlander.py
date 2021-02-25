import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from agent import DQNAgent, DuelingDQNAgent, PrioritizedDQNAgent
from experiment import Experiment
from logger import get_logger

number_of_episodes = 3000
show_every = 100  # Number of episodes.
test_every = 100  # Number of episodes.


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")

    # Create a new agent here.
    experiment = Experiment.create(
        base_name="dqn/dqn_LunarLander-v2",
        model_class=DQNAgent,
        hp={
            "env": env,
            "loss_function": torch.nn.SmoothL1Loss(),
            "memory_size": 10000,
            "batch_size": 64,
            "epsilon_0": 0.1,
            "gamma": 0.999,
            "lr": 0.001,
            "q_layers": [32, 32],
            "sync_frequency": 1000,
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
        show = (experiment.episode + 1) % show_every == 0
        test = (experiment.episode + 1) % test_every == 0

        # Run the episode.
        observation = experiment.model.env.reset()
        episode_loss, episode_reward, episode_steps = 0, 0, 0

        while True:
            episode_steps += 1

            if test:
                experiment.model.env.render()
                observation, reward, done, info = experiment.model.env.step(
                    experiment.model.step_test(observation)
                )
            else:
                action, loss = experiment.model.step_train(observation)
                observation, reward, done, info = experiment.model.env.step(action)

                experiment.model.record(
                    observation,
                    reward,
                    False
                    if info.get("TimeLimit.truncated")
                    else done,  # For CartPole: not a "real" done.
                )

                if loss is not None:
                    episode_loss += float(loss)

            episode_reward += reward

            if done:
                experiment.episode += 1
                break

        # Logging.
        if test:
            logger.info(
                f"Test {experiment.episode // test_every}: reward = {episode_reward}"
            )

            writer.add_scalars(
                "test",
                {"reward": episode_reward, "steps": episode_steps},
                global_step=experiment.episode,
            )
        else:
            writer.add_scalars(
                "train",
                {
                    "loss": episode_loss / episode_steps,
                    "reward": episode_reward,
                    "steps": episode_steps,
                },
                global_step=experiment.episode,
            )

        if show:
            logger.info(f"Info episode {experiment.episode}")
            logger.info(f"\tMemory length: {len(experiment.model.memory)}")
            logger.info(
                f"\tTrain steps since last sync: {experiment.model.train_steps_since_last_sync}"
            )
            logger.info(
                f"\tepsilon = {1 / (1 + experiment.model.epsilon_0 * experiment.model.train_steps)}"
            )

    env.close()
