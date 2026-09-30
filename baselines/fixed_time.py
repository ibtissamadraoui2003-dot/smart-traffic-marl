"""Controleur "feux fixes" : alterne les phases a intervalle regulier,
sans jamais regarder l'etat du trafic. C'est la reference classique
utilisee dans la litterature pour mesurer l'apport d'un controleur adaptatif.
"""

from constants import FIXED_TIME_INTERVAL, PHASE_NS, PHASE_EW


class FixedTimeController:
    def __init__(self, interval=FIXED_TIME_INTERVAL):
        self.interval = interval

    def act(self, state, step_count):
        """Meme phase pour tous les carrefours, basee uniquement sur l'horloge."""
        cycle = (step_count // self.interval) % 2
        return PHASE_NS if cycle == 0 else PHASE_EW
