# VAE : Variational Auto-Encoder



On implémente dans ce TP un modèle génératif par auto-encodage variationnel. On met en place un modèle linéaire et un convolutionnel pour étudier les différences produites. 

## Entrainement

Les modèles produits apprennent correctement. On peut voir dans la figure ci-dessous la descente de gradient pour :

- un modèle linéaire de dimension latente 2 (rouge)
- un modèle linéaire de dimension latente 10 (cyan)
- un modèle convolutionnel de dimension latente 2 (bleu)
- un modèle convolutionnel de dimension latente 10 (rose)



<div align="center">   
	<img src="images/train_loss.svg" width="450"/> 
  <figcaption>Train loss</figcaption>
</div>

Une dimension plus grande permet un meilleur apprentissage du modèle, laissant plus de fléxibilité et de détaille dans la représentation latente. Par ailleurs, les réseaux convolutionnels permettent d'amélioré la proximité avec la distribution latente normale et la reconstritution de l'image de départ.

On se concentrera ainsi sur les réseaux convolutionnels offrant un meilleur apprentissage (une fonction de loss plus faible en test).

## Évaluation

<div align="center">   
	<img src="images/conv2/original.png" width="450"/> 
  <figcaption>Images originales</figcaption>
	<img src="images/conv2/reconstructed.png" width="450"/> 
  <figcaption>Reconstitution (convolution, dim 2)</figcaption>
	<img src="images/conv10/reconstitution.png" width="450"/> 
  <figcaption>Reconstitution (convolution, dim 10)</figcaption>
</div>

On peut apprécier sur la figure ci-dessus que les réseaux reconstituent correctement les images de la base de données. Par ailleurs, le modèle utilisant une dimension de l'espace latent de 10 est plus précis, plus net. 

Cele se remarque dans les images générées alétoirement à partir d'une distribution normale sur l'espace latent :

<div align="center">   
	<img src="images/conv2/generated.png" width="450"/> 
  <figcaption>Génération (convolution, dim 2)</figcaption>
	<img src="images/conv10/generated.png" width="450"/> 
  <figcaption>Génération (convolution, dim 10)</figcaption>
</div>

Les images sont plus nettes avec un espace plus grand. Néanmoins, elles ne paraissent pas forcément plus vraisemblables. En effet, un plus gros contraste sur le modèle de dimension 10 accentue certains défauts.

Nous avons tenter de mettre en place une interpolation entre deux points de l'espace latent. Ce test permet de vérifier si le prior gaussien a bien permis un régularisation de l'espace latent.

<div align="center">   
	<img src="images/conv2/interpolation.png" width="450"/> 
  <figcaption>Interpolation (convolution, dim 2)</figcaption>
	<img src="images/conv10/interpolation.png" width="450"/> 
  <figcaption>Interpolation (convolution, dim 10)</figcaption>
</div>

Il apparait que l'interpolation du modèle de dimension 2 passe toujours par des données vraisemblables (1, 9, et 7). Cela est moins flagrant pour le modèle de dimension 10 où les chiffres sont plus difficiles à distinguer. 



Finalement, on génère des images depuis une grille de données latentes $[-1.5, 1.5]^2$. On retrouve une continuité dans les données produites. Par ailleurs, il est possible de visualisé l'interpolation produite plus haut. Finalement, le 8 semble jouer un rôle centrale dans cette distribution. Cela semble compréhensible au vu de sa géométrie caligraphique occupant un grand espace. 

![map](/Users/victorduthoit/Documents/cours/3A/Master DAC/RLD/rl-exercices/TME9/images/conv2/map.png)

## Conclusion

Pour conclure, on a vu qu'il était capable de faire apprendre une distribution normale à des données par encodage. Si les modèles linéaires apprennent correctement, leurs homologues convolutionnel semblent toutefois les surpasser. Le choix de la dimension latente n'est pas évident. Si des modèles à dimension plus élevé permettent une meilleur dminution de l'erreur, on peut appercevoir des abérations trop contrastées lors de la génération. Un modèle à faible dimension n'offre pas autant de contraste lors de la reconsitution mais cela lui permet de ne pas accentuer ses défaults lors de la génération. 