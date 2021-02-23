import gym
from torch.utils.tensorboard import SummaryWriter

from agent.q_learning import *
from experiment import Experiment
import gridworld
from logger import get_logger

number_of_episodes = 3000
max_steps = 1000
show_every = 1000  # Number of episodes.
pause = 0.1
plan = 3

if __name__ == "__main__":
    env = gym.make("gridworld-v0")
    env.setPlan(f"gridworldPlans/plan{plan}.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    obs = env.reset()

    # Create a new agent here.
    experiment = Experiment.create(
        base_name=f"q_learning/q_learning_decay_gridworld-v0-plan{plan}",
        model_class=QLearning,
        hp={
            "env": env,
            "action_space": range(env.action_space.n),
            "observation_shape": obs,
            # "action_strategy": EpsilonGreedy(epsilon=0.1),
            "action_strategy": EpsilonGreedyDecay(epsilon=0.1, alpha=1000),
            "update_strategy": OfflinePolicy(alpha=0.1, gamma=0.99),
            "action_to_str": {0: "south", 1: "north", 2: "west", 3: "east"},
        },
    )
    experiment.save()

    logger = get_logger(experiment.name, file_path=experiment.log_path)
    writer = SummaryWriter(
        log_dir=experiment.writer_path, purge_step=experiment.episode
    )
    experiment.info(logger)

    while experiment.episode < number_of_episodes:
        show = (experiment.episode + 1) % show_every == 0

        # Reset the environment and the agent.
        experiment.model.reset()
        done = False
        episode_reward = 0

        while experiment.model.t < max_steps and not done:
            if show:
                env.render(pause=pause)

            reward, done, _ = experiment.model.step()
            episode_reward += reward

        experiment.episode += 1

        # Log.
        if show:
            logger.info(f"Episode {experiment.episode}: reward = {episode_reward}.")

        writer.add_scalars(
            "train",
            {"reward": episode_reward, "steps": experiment.model.t},
            global_step=experiment.episode,
        )

    env.close()
