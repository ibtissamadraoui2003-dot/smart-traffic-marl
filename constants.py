"""
Constantes partagees par tout le projet.
Regroupees ici pour que l'environnement, les agents et les baselines
utilisent toujours les memes valeurs.
"""

# ----------------------------------------------------------------------
# Grille / simulation
# ----------------------------------------------------------------------
GRID_SIZE = 2            # grille GRID_SIZE x GRID_SIZE d'intersections
MAX_QUEUE = 20            # capacite max d'une file (vehicules)
FLOW_RATE = 2              # vehicules qui passent par direction verte et par pas
EPISODE_LENGTH = 300      # pas de temps par episode

# Generation de trafic (spawn aux bords de la grille)
SPAWN_PROB_BASE = 0.35     # proba d'un nouveau vehicule par pas, hors heure de pointe
SPAWN_PROB_RUSH = 0.65     # proba pendant l'heure de pointe
RUSH_HOUR_START = 100       # debut heure de pointe (pas de temps dans l'episode)
RUSH_HOUR_END = 200         # fin heure de pointe

# Penalite appliquee quand un agent change de phase (evite le flicker,
# approxime le cout de securite d'un vrai feu qui doit passer par l'orange)
SWITCH_PENALTY = 1.0

# ----------------------------------------------------------------------
# Discretisation de l'etat (pour le Q-Learning tabulaire)
# ----------------------------------------------------------------------
QUEUE_BUCKETS = 5          # nombre de niveaux de congestion (0..4)
BUCKET_SIZE = MAX_QUEUE / QUEUE_BUCKETS

# Phases : 0 = axe Nord-Sud vert, 1 = axe Est-Ouest vert
PHASE_NS = 0
PHASE_EW = 1
PHASE_DIRECTIONS = {
    PHASE_NS: ("N", "S"),
    PHASE_EW: ("E", "W"),
}
DIRECTIONS = ("N", "S", "E", "W")

# ----------------------------------------------------------------------
# Q-Learning
# ----------------------------------------------------------------------
ALPHA = 0.2                 # taux d'apprentissage
GAMMA = 0.9                 # facteur d'actualisation
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.998        # decroissance appliquee une fois par episode

NUM_EPISODES = 2000
BETA_COORDINATION = 0.5       # poids de la recompense des voisins (mode coordonne)

# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------
FIXED_TIME_INTERVAL = 30      # pas de temps entre deux changements de phase


def bucket(value, max_value=MAX_QUEUE, buckets=QUEUE_BUCKETS):
    """Discretise une file d'attente en `buckets` niveaux (0 .. buckets-1)."""
    step = max_value / buckets
    b = int(value // step)
    return min(b, buckets - 1)
