# TME 3 — Q-Learning

_Victor Duthoit, Pierre Wan-Fat_

Dans ce TME, on utilise l’environnement GridWorld. Les cartes 2 et 3 ont été utilisées, les cartes précédentes étant trop simples et la carte 4 étant déjà trop grosse pour que l’entraînement se finisse en un temps raisonnable. Les courbes présentées représentent 3 000 épisodes.

On a implémenté les algorithmes Q-Learning et SARSA, avec deux types d’exploration (*epsilon greedy* et *epsilon greedy decay*).

## Hyperparamètres

Ces hyperparamètres sont utilisés pour les deux environnements.

```python
# Q-Learning.
	alpha = 0.1
	gamma = 0.99

# epsilon greedy.
	epsilon = 0.1

# epsilon greedy decay (epsilon = epsilon_0 * alpha / global_step)
    epsilon_0 = 0.1
    alpha = 1000
```

## Plan 2

<img src="tme_3/plan2.png" style="zoom:60%;" />

On remarque la présence d’un bloc jaune qui récompense beaucoup, mais qui est placé après le bloc vert (terminal) sur le chemin naturel de l’agent.

### Q-Learning

Avec une exploration *epsilon greedy*, on voit que l’apprentissage se fait très rapidement (convergence après 200 épisodes) mais l’agent ne parvient presque jamais à aller prendre le bloc jaune.

![](tme_3/q_learning_plan2.svg)

L’exploration *epsilon greedy decay* améliore les performances de l’agent, qui parvient à récupérer la récompense additionnelle beaucoup plus souvent.

![](tme_3/q_learning_decay_plan2.svg)

### SARSA

SARSA ne parvient en revanche jamais à atteindre la récompense additionnelle, que ce soit avec la stratégie *epsilon greedy* classique ·

![](tme_3/sarsa_plan2.svg)

Ou avec la stratégie *epsilon greedy decay*. On remarque cependant que les performances oscillent moins.

![](tme_3/sarsa_decay_plan2.svg)

## Plan 3

<img src="tme_3/plan3.png" style="zoom:60%;" />

Deux blocs verts sont présents, l’un étant légèrement plus accessible que l’autre.

### Q-Learning

Avec une exploration *epsilon greedy*, l’apprentissage se fait rapidement, et les oscillations sont assez modérées.

![](tme_3/q_learning_plan3.svg)

L’exploration *epsilon greedy decay* améliore les performances de l’agent après convergence, même si l’apprentissage prend plus de temps.

![](tme_3/q_learning_decay_plan3.svg)

### SARSA

SARSA a des bonnes performances et une convergence rapide :

![](tme_3/sarsa_plan3.svg)

La stratégie *epsilon greedy decay* prend un peu plus de temps mais oscille moins après convergence :

![](tme_3/sarsa_decay_plan3.svg)