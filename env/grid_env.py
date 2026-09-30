"""
Environnement de simulation de trafic multi-agents.

Une grille GRID_SIZE x GRID_SIZE de carrefours. Chaque carrefour a 4 files
d'attente (N, S, E, W) representant les vehicules arrivant de chaque
direction. Un vehicule dans la file "N" est arrive du nord et se dirige
vers le sud ; quand il traverse le carrefour, il rejoint la file "N" du
carrefour situe juste au sud (ou sort de la grille si on est deja au bord).

Simplifications assumees (documentees dans docs/EXPLICATION.md) :
- pas de gestion des tourne-a-gauche/droite (uniquement le trafic tout droit)
- pas de phase orange explicite (une penalite de changement la remplace)
- pas de blocage amont (spillback) : une file pleine absorbe simplement
  moins de nouveaux vehicules, elle ne bloque pas le carrefour precedent
- mise a jour synchrone : tous les departs sont calcules a partir de l'etat
  au debut du pas, puis appliques ensemble (evite les effets d'ordre)
"""

import random

from constants import (
    GRID_SIZE, MAX_QUEUE, FLOW_RATE, EPISODE_LENGTH,
    SPAWN_PROB_BASE, SPAWN_PROB_RUSH, RUSH_HOUR_START, RUSH_HOUR_END,
    SWITCH_PENALTY, PHASE_NS, PHASE_EW, PHASE_DIRECTIONS, DIRECTIONS,
    bucket,
)

# Pour une direction donnee, le carrefour voisin vers lequel un vehicule
# se dirige en sortant du carrefour courant (dr, dc a ajouter a (r, c)).
_TARGET_OFFSET = {
    "N": (1, 0),    # vient du nord, continue vers le sud
    "S": (-1, 0),   # vient du sud, continue vers le nord
    "E": (0, -1),   # vient de l'est, continue vers l'ouest
    "W": (0, 1),    # vient de l'ouest, continue vers l'est
}


class TrafficGridEnv:
    """Environnement multi-agent : un agent = un carrefour."""

    def __init__(self, grid_size=GRID_SIZE, episode_length=EPISODE_LENGTH, seed=None):
        self.grid_size = grid_size
        self.episode_length = episode_length
        self.rng = random.Random(seed)

        self.intersections = [
            (r, c) for r in range(grid_size) for c in range(grid_size)
        ]

        self.queues = {}
        self.phases = {}
        self.step_count = 0
        self.total_exits = 0
        self.total_spawned = 0

        self._neighbors_cache = {
            i: self._compute_neighbors(i) for i in self.intersections
        }

        self.reset()

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------
    def reset(self):
        self.queues = {
            i: {d: 0 for d in DIRECTIONS} for i in self.intersections
        }
        self.phases = {i: PHASE_NS for i in self.intersections}
        self.step_count = 0
        self.total_exits = 0
        self.total_spawned = 0
        return self._all_states()

    def step(self, actions):
        """
        actions : dict {(r, c): 0 ou 1} -- phase choisie par chaque agent
        retourne (next_states, local_rewards, done, info)
        """
        # 1) recompense basee sur l'etat AVANT ce pas (vehicules deja en attente)
        old_queue_sum = {
            i: sum(self.queues[i].values()) for i in self.intersections
        }
        switched = {
            i: (actions[i] != self.phases[i]) for i in self.intersections
        }

        # 2) appliquer la nouvelle phase choisie par chaque agent
        for i in self.intersections:
            self.phases[i] = actions[i]

        # 3) calculer tous les departs a partir de l'etat courant (synchrone)
        pending_arrivals = []  # (cible, direction, nb_vehicules)
        step_exits = 0
        for i in self.intersections:
            released_dirs = PHASE_DIRECTIONS[self.phases[i]]
            for d in released_dirs:
                depart = min(FLOW_RATE, self.queues[i][d])
                if depart == 0:
                    continue
                self.queues[i][d] -= depart
                target = self._shift(i, d)
                if target in self.queues:
                    pending_arrivals.append((target, d, depart))
                else:
                    step_exits += depart

        # 4) appliquer les arrivees (bufferisees) en respectant la capacite max
        for target, d, depart in pending_arrivals:
            self.queues[target][d] = min(MAX_QUEUE, self.queues[target][d] + depart)

        # 5) generation de nouveaux vehicules aux bords de la grille
        rush_hour = RUSH_HOUR_START <= self.step_count < RUSH_HOUR_END
        spawn_prob = SPAWN_PROB_RUSH if rush_hour else SPAWN_PROB_BASE
        step_spawned = 0
        for i in self.intersections:
            for d in DIRECTIONS:
                if self._shift(i, d, upstream=True) not in self.queues:
                    if self.rng.random() < spawn_prob:
                        before = self.queues[i][d]
                        self.queues[i][d] = min(MAX_QUEUE, before + 1)
                        step_spawned += self.queues[i][d] - before

        self.total_exits += step_exits
        self.total_spawned += step_spawned

        # 6) recompenses locales
        local_rewards = {}
        for i in self.intersections:
            r = -float(old_queue_sum[i])
            if switched[i]:
                r -= SWITCH_PENALTY
            local_rewards[i] = r

        self.step_count += 1
        done = self.step_count >= self.episode_length

        next_states = self._all_states()
        info = {
            "total_queue": sum(old_queue_sum.values()),
            "exits_this_step": step_exits,
            "spawned_this_step": step_spawned,
            "rush_hour": rush_hour,
        }
        return next_states, local_rewards, done, info

    def get_neighbors(self, intersection):
        """Carrefours adjacents existants (haut/bas/gauche/droite)."""
        return self._neighbors_cache[intersection]

    def get_max_queue_snapshot(self):
        """Renvoie la file la plus longue de toute la grille (indicateur d'equite)."""
        return max(sum(q.values()) for q in self.queues.values())

    # ------------------------------------------------------------------
    # Aides internes
    # ------------------------------------------------------------------
    def _all_states(self):
        return {i: self._state_of(i) for i in self.intersections}

    def _state_of(self, i):
        q = self.queues[i]
        return (
            bucket(q["N"]), bucket(q["S"]), bucket(q["E"]), bucket(q["W"]),
            self.phases[i],
        )

    def _shift(self, i, direction, upstream=False):
        """
        Carrefour voisin dans la direction donnee.
        upstream=False -> carrefour EN AVAL (ou va un vehicule qui part)
        upstream=True  -> carrefour EN AMONT (qui alimente cette file)
        Les deux sont symetriques ici car le decalage est le meme axe.
        """
        r, c = i
        dr, dc = _TARGET_OFFSET[direction]
        if upstream:
            dr, dc = -dr, -dc
        return (r + dr, c + dc)

    def _compute_neighbors(self, i):
        r, c = i
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [n for n in candidates if n in set(self.intersections)]
