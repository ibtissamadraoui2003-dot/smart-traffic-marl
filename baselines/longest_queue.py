"""Controleur heuristique : donne le vert a l'axe (Nord-Sud ou Est-Ouest)
qui a la file cumulee la plus longue. Pas d'apprentissage, mais reactif
au trafic reel -- une reference plus forte que les feux fixes.
"""

from constants import PHASE_NS, PHASE_EW


class LongestQueueController:
    def act(self, state, step_count=None):
        bucket_n, bucket_s, bucket_e, bucket_w, _phase = state
        ns_load = bucket_n + bucket_s
        ew_load = bucket_e + bucket_w
        return PHASE_NS if ns_load >= ew_load else PHASE_EW
