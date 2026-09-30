"""
Agent Q-Learning tabulaire pour un carrefour.

Chaque agent maintient sa propre table Q(s, a) -> valeur, exactement comme
dans le projet Ball Sort IQL (apprentissage independant). Ce qui change
d'un carrefour a l'autre, c'est uniquement la recompense qu'on lui donne :
- mode "local"       : uniquement sa propre congestion
- mode "coordonne"   : sa congestion + celle de ses voisins (voir train.py)
"""

import pickle
import random
from collections import defaultdict

from constants import ALPHA, GAMMA, EPSILON_START, EPSILON_MIN, EPSILON_DECAY

ACTIONS = (0, 1)  # 0 = phase Nord-Sud verte, 1 = phase Est-Ouest verte


def _default_q_values():
    """Fonction top-level (et non lambda) pour rester picklable."""
    return {a: 0.0 for a in ACTIONS}


class QLearningAgent:
    def __init__(self, agent_id, alpha=ALPHA, gamma=GAMMA,
                 epsilon=EPSILON_START, epsilon_min=EPSILON_MIN,
                 epsilon_decay=EPSILON_DECAY, rng=None):
        self.agent_id = agent_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = rng or random.Random()
        self.q_table = defaultdict(_default_q_values)

    def select_action(self, state, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.choice(ACTIONS)
        q_values = self.q_table[state]
        best_value = max(q_values.values())
        best_actions = [a for a, v in q_values.items() if v == best_value]
        return self.rng.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        best_next = 0.0 if done else max(self.q_table[next_state].values())
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(_default_q_values)
        self.q_table.update(data)

    def __len__(self):
        """Nombre d'etats distincts rencontres (utile pour verifier l'apprentissage)."""
        return len(self.q_table)
