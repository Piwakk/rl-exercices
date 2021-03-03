# TME RLADL

Answers to the Reinforcement Learning course exercises of [Sorbonne Université M2 DAC](http://dac.lip6.fr/master/) (Master of Data Science),
by Victor Duthoit and Pierre Wan-Fat.

## Useful resources

- [The course page](http://dac.lip6.fr/master/rld-2020-2021/)
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/)
- https://github.com/seungeunrho/minimalRL/

## Lessons learned

- Learn as soon as possible how to use TensorBoard and the logging module, and implement a robust checkpointing module.
  This will save you a lot of time.
- Start simple: you don't need to code advanced techniques (such as Target Network or Prioritized Replay Buffer) at first,
  especially on simple environments (CartPole).
- As soon as you are quite confident in your algorithm, use a grid search to tune the hyperparameters and launch it on the PPTI.
  Don't try to tune the hyperparameters by yourself, this will likely not work, and you will lose a lot of time as well as your mind!
- Your teachers will sometimes give you a LOT of (confusing) boilerplate code. You don't have to use it to succeed; starting fresh
  is sometimes the best choice.

### Using the PPTI

- Before doing anything, check that nobody is already using the machine you are connected to (`who` and `nvidia-smi`).
- There is always a risk that your script randomly crashes without you noticing it, and anyways the PPTI reboots every day at 8 AM.
  So be sure to log and checkpoint everything you do.
- Learn how to use `tmux`.
- The magic command to install a Python package is `pip3 install --user --proxy=proxy:3128 xxx`.
- Git also works on the PPTI, you just need to configure the HTTP proxy to `proxy:3128`.
- Use `/tempory` if you don't have enough space in your home folder (which is limited to around 3 GB).

## Implemented algorithms

- TME 1:
  - Epsilon greedy, UCB and Lin-UCB for Bandits.
- TME 2:
  - Policy Iteration on GridWorld.
  - Value Iteration on GridWorld.
- TME 3:
  - QLearning on Gridworld.
  - SARSA on Gridworld.
- TME 4:
  - DQN on CartPole.
  - Dueling DQN on CartPole.
  - Prioritized DQN on CartPole.
  - DQN on LunarLander.
- TME 5: Actor-Critic.
  - TD(0) on CartPole.
  - TD(0) on LunarLander.
- TME 6: PPO.
  - Adaptive PPO on CartPole.
  - Adaptive PPO on LunarLander.
  - Clipped PPO on CartPole.
  - Clipped PPO on LunarLander.
- TME 7: DDPG.
  - DDPG on Pendulum.
  - DDPG on LunarLander.
  - DDPG on MountainCar.
- TME 8: GAN.
- TME 9: VAE
  - VAE (Linear).
  - VAE (Convolutional).
- TME 10: MADDPG
  - Simple spread.
  - Simple adversary.
  - Simple tag.
- TME 11: Imitation learning
  - Behavior cloning on LunarLander.
  - GAIL on LunarLander.
- TME 12: Curriculum Learning
  - Goal sampling.
  - HER (Hinsight Experience Replay).
  - ISG (Iterative Goal Sampling).
