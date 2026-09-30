# Régulation de trafic multi-agents : explication complète

## 1. Le problème

Un reseau de carrefours doit ecouler du trafic le plus efficacement possible.
Des feux a minuterie fixe ne reagissent jamais a la realite du trafic : ils
restent verts meme quand aucune voiture n'attend, et restent rouges meme
quand une file s'accumule. Un controleur adaptatif -- qui observe les files
d'attente et choisit la phase en consequence -- peut faire beaucoup mieux.

Mais un carrefour isole qui optimise seulement SA PROPRE file peut, sans le
vouloir, deverser plus de trafic sur le carrefour voisin et deplacer le
probleme au lieu de le resoudre. C'est exactement le genre de probleme de
coordination multi-agents etudie dans ce module de formation.

## 2. Modelisation de l'environnement

### Grille et files d'attente

Le reseau est une grille `GRID_SIZE x GRID_SIZE` de carrefours. Chaque
carrefour a 4 files d'attente : **N** (vehicules arrives du nord, en
direction du sud), **S** (arrives du sud, vers le nord), **E** (arrives de
l'est, vers l'ouest) et **W** (arrives de l'ouest, vers l'est).

Quand un vehicule traverse un carrefour, il rejoint la file de MEME
direction du carrefour suivant (une file "N" reste une file "N" tout le
long de son trajet), ou sort de la grille s'il n'y a plus de carrefour
suivant dans cette direction.

### Etat observe par un agent

Chaque agent (= un carrefour) observe uniquement son propre etat local :

```
(bucket(N), bucket(S), bucket(E), bucket(W), phase_actuelle)
```

Les files sont discretisees en 5 niveaux de congestion (`bucket()` dans
`constants.py`) pour garder une table Q de taille raisonnable
(5^4 x 2 = 1250 etats possibles par agent).

### Actions

Deux phases possibles :
- **Phase 0** : axe Nord-Sud vert (les files N et S peuvent avancer)
- **Phase 1** : axe Est-Ouest vert (les files E et W peuvent avancer)

L'agent choisit une phase a chaque pas de temps. Changer de phase entraine
une petite penalite (`SWITCH_PENALTY`) pour eviter un comportement instable
qui alternerait en permanence -- une approximation simple du temps de
securite (orange) d'un vrai feu, sans avoir a modeliser une 3e phase.

### Recompense

`recompense = -(somme des 4 files d'attente au debut du pas) - penalite_si_changement`

Un agent qui minimise cette recompense minimise donc la congestion
accumulee a son carrefour -- un proxy standard du temps d'attente total
(base sur la loi de Little : le temps d'attente cumule est proportionnel a
la longueur de la file integree dans le temps).

## 3. L'algorithme : Q-Learning tabulaire

Exactement le meme algorithme que Ball Sort IQL :

```
Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
```

Chaque carrefour a sa propre table Q, remplie independamment (Independent
Q-Learning). Politique epsilon-greedy pendant l'entrainement, purement
gloutonne (epsilon=0) a l'evaluation.

| Parametre | Valeur |
|---|---|
| alpha (taux d'apprentissage) | 0.2 |
| gamma (facteur d'actualisation) | 0.9 |
| epsilon initial | 1.0 |
| epsilon minimum | 0.05 |
| decroissance de epsilon | 0.998 par episode |
| episodes d'entrainement | 2000 |
| longueur d'un episode | 300 pas |

## 4. Deux modes de recompense : local vs coordonne

C'est le coeur experimental du projet, et il prolonge directement les deux
projets precedents de la formation :

- **Minimax Pong** : deux agents en conflit total (jeu a somme nulle).
- **Ball Sort IQL** : agents independants, recompense **partagee**
  (cooperation implicite).
- **Ce projet** : on peut choisir, PAR AGENT, entre :
  - **mode `local`** : recompense = uniquement sa propre congestion
    (comportement egoiste)
  - **mode `coordonne`** : recompense = sa congestion + `BETA` fois la
    congestion moyenne de ses voisins directs (`BETA_COORDINATION = 0.5`)

Le mode coordonne est calcule dans `train.py::compute_rewards()`, pas dans
l'environnement : l'environnement renvoie toujours les recompenses locales
brutes, et c'est la boucle d'entrainement qui les combine si necessaire.
Cela permet de reutiliser exactement le meme environnement et le meme agent
pour les deux experiences.

## 5. Les baselines (references de comparaison)

Sans point de comparaison, impossible de prouver qu'un controleur "marche
bien". Deux baselines sans apprentissage :

- **Feux fixes** (`baselines/fixed_time.py`) : alterne les phases toutes
  les 30 pas, sans jamais regarder le trafic. C'est la reference "aucune
  IA", equivalent a la plupart des feux reels aujourd'hui.
- **File la plus longue** (`baselines/longest_queue.py`) : donne le vert a
  l'axe le plus charge. Reactif, mais sans memoire ni apprentissage.

## 6. Resultats obtenus

Evaluation sur 30 episodes, grille 2x2, avec les 4 controleurs (voir
`results/comparison.csv` et `results/comparison_queue.png`) :

| Controleur | File moyenne | Debit | Pic max |
|---|---|---|---|
| Feux fixes | 43.51 | 1021.2 | 23.63 |
| File la plus longue | 21.36 | 1059.2 | 10.87 |
| Q-Learning local | **18.54** | 1063.8 | 11.53 |
| Q-Learning coordonne | 18.88 | 1063.4 | 13.27 |

**Le Q-Learning local reduit la congestion de 57 % par rapport aux feux
fixes**, et bat meme l'heuristique "file la plus longue" de 13 %, alors
qu'il n'a acces qu'a son propre etat et n'a jamais recu la moindre regle
ecrite a la main -- tout est appris par essai-erreur.

### Un resultat honnete et interessant : la coordination n'aide pas ici

Sur cette grille 2x2, le mode `coordonne` fait legerement **moins bien**
que le mode `local` (18.88 contre 18.54). Ce n'est pas un bug : voir
`results/learning_curves.png`, la courbe coordonnee est clairement moins
stable pendant l'entrainement.

L'explication la plus probable : sur une grille aussi petite, chaque
carrefour n'a que 2 voisins, et leur congestion est deja fortement
correlee a la sienne. Ajouter leur recompense au signal d'apprentissage
n'apporte donc pas d'information nouvelle, mais ajoute du bruit et de la
**non-stationnarite** -- exactement la limite deja identifiee dans le
projet Ball Sort IQL : chaque agent apprend en considerant les autres
comme faisant partie d'un environnement qui change sans cesse.

**Hypothese testable** : sur une grille plus grande (3x3 ou 4x4), ou un
embouteillage a un coin peut se propager en cascade sur plusieurs
carrefours avant de revenir, la coordination devrait apporter un gain
mesurable la ou l'egoisme local echoue. C'est la piste d'amelioration la
plus interessante du projet (voir `--grid-size 3`).

## 7. Simplifications assumees

Pour rester robuste et livrable sans bug, ce projet fait des choix
deliberement simples :

- **Pas de virages** : uniquement du trafic tout droit (pas de gestion des
  tourne-a-gauche/droite).
- **Pas de phase orange explicite** : remplacee par une penalite de
  changement de phase.
- **Pas de blocage amont (spillback)** : une file pleine absorbe moins de
  nouveaux vehicules mais ne bloque jamais physiquement le carrefour
  precedent.
- **Mise a jour synchrone** : tous les departs sont calcules a partir de
  l'etat au debut du pas, puis appliques ensemble.

Ces simplifications sont documentees ici volontairement : un vrai projet
de recherche assume et explique ses limites plutot que de les cacher.

## 8. Pistes d'amelioration

- Tester `--grid-size 3` ou `4` pour verifier l'hypothese de la section 6.
- Remplacer le Q-Learning tabulaire par un DQN (reseau de neurones) pour
  passer a l'echelle sur de grandes grilles.
- Ajouter une vraie communication entre agents voisins (message passing)
  plutot qu'un simple melange de recompenses.
- Modeliser les virages et une vraie phase orange.
- Utiliser des donnees de trafic reelles (comptages routiers publics) pour
  calibrer les taux d'arrivee des vehicules.

## References

- Littman, M. (1994). *Markov games as a framework for multi-agent
  reinforcement learning.*
- Sutton, R. & Barto, A. *Reinforcement Learning: An Introduction.*
- Wei, H. et al. *IntelliLight / PressLight* -- travaux de reference sur le
  controle de trafic par apprentissage par renforcement.
- SUMO (Simulation of Urban MObility) -- simulateur de reference utilise en
  recherche, alternative possible a l'environnement simplifie de ce projet.
