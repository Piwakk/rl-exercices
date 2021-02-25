# Etat

| TP                 | État                    |        |
| ------------------ | ----------------------- | ------ |
| 1 (Bandits)        | :white_check_mark:      |        |
| 2 (Value / Policy) | :white_check_mark:      |        |
| 3 (Q-Learning)     | :white_check_mark:      |        |
| 4 (DQN)            | Rapport - pdf 1         | Pierre |
| 5 (Actor-Critic)   | Rapport - pdf 1         | Pierre |
| 6 (PPO)            | Rapport - pdf 1         | Pierre |
| 7 (DDPG)           | Rapport - pdf 1         | Pierre |
| 8 (GAN)            | faire + Rapport - pdf 2 |        |
| 9 (VAE)            | faire + Rapport - pdf 2 |        |
| 10 (MADDPG)        | Rapport - pdf 3         | victor |
| 11 (GAIL)          | faire + Rapport - pdf 3 | victor |
| 12 (Curriculum)    | faire + Rapport - pdf 3 |        |



hyperparamètres

| agent        | Environnement           | Hyperparamètres                                                      | Courbes |
| ------------ | ----------------------- | -------------------------------------------------------------------- | ------- |
| DQN          | Cart pole               | dqn\_-h[128, 128]\_lr0.0005_g0.98_eps01.0_nu0.1_clear20              |         |
|              | Lunar Lander            | dqn\_-h[128, 128]\_lr1e-07_g0.98_eps01.0_nu0.1_clear20               |         |
|              | Gridworld v0            | dqn\_-h[128, 128]\_lr1e-06_g0.98_eps01.0_nu0.1_clear20-              |         |
| Actor Critic | Cart pole               | h200_lr0.01_g0.98_clear10                                            |         |
|              | Lunar Lander            | Pas de bon                                                           |         |
|              | Gridworld v0            | Aucun                                                                |         |
| PPO          | Cart pole               | h300_lr0.001_g0.98_delta0.8_beta0.001                                |         |
|              | Lunar Lander            | Pas de bon                                                           |         |
|              | Gridworld v0            | Aucun                                                                |         |
| DDPG         | Pendulum                | \_h200_lrmu0.005_lrq0.001_g0.98_K10_rho0.995_mb100                   |         |
|              | Lunar Lander Continuous | \_h100_lrmu0.0001_lrq0.001_g0.98_K10_rho0.9_mb100 -> pas terrible... |         |
|              | Mountain car            | \_h200_lrmu0.005_lrq0.0001_g0.98_K10_rho0.995_mb100 -> pas terrible  |         |
