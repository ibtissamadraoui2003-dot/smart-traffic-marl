"""
Entraine une equipe d'agents Q-Learning (un agent par carrefour) sur
l'environnement de trafic.

Deux modes de recompense, comparables directement dans evaluate.py :
- local       : chaque agent minimise UNIQUEMENT sa propre congestion
                (equivalent egoiste, proche de plusieurs agents Minimax
                independants qui ne cooperent pas)
- coordonne   : chaque agent minimise sa congestion + celle de ses voisins
                (equivalent a la recompense partagee du projet Ball Sort IQL)

Usage :
    python train.py --mode local
    python train.py --mode coordonne --episodes 3000 --grid-size 2
"""

import argparse
import csv
import json
import os
import random
import time

from constants import NUM_EPISODES, BETA_COORDINATION, GRID_SIZE
from env.grid_env import TrafficGridEnv
from agents.q_learning_agent import QLearningAgent


def compute_rewards(mode, local_rewards, neighbors_map, beta=BETA_COORDINATION):
    if mode == "local":
        return local_rewards
    rewards = {}
    for i, r in local_rewards.items():
        nbrs = neighbors_map[i]
        nbr_avg = sum(local_rewards[n] for n in nbrs) / len(nbrs) if nbrs else 0.0
        rewards[i] = r + beta * nbr_avg
    return rewards


def train(mode, episodes, grid_size, seed, output_dir, log_every=100, quiet=False):
    assert mode in ("local", "coordonne"), "mode doit etre 'local' ou 'coordonne'"

    rng = random.Random(seed)
    env = TrafficGridEnv(grid_size=grid_size, seed=seed)
    neighbors_map = {i: env.get_neighbors(i) for i in env.intersections}
    agents = {
        i: QLearningAgent(i, rng=random.Random(rng.randint(0, 10_000)))
        for i in env.intersections
    }

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"train_log_{mode}.csv")
    log_rows = []

    start = time.time()
    for episode in range(1, episodes + 1):
        states = env.reset()
        episode_queue_sum = 0
        episode_exits = 0

        for _step in range(env.episode_length):
            actions = {i: agents[i].select_action(states[i]) for i in env.intersections}
            next_states, local_rewards, done, info = env.step(actions)
            rewards = compute_rewards(mode, local_rewards, neighbors_map)

            for i in env.intersections:
                agents[i].update(states[i], actions[i], rewards[i], next_states[i], done)

            episode_queue_sum += info["total_queue"]
            episode_exits += info["exits_this_step"]
            states = next_states

        for i in env.intersections:
            agents[i].decay_epsilon()

        avg_queue = episode_queue_sum / env.episode_length
        log_rows.append({
            "episode": episode,
            "avg_queue": round(avg_queue, 3),
            "total_exits": episode_exits,
            "epsilon": round(agents[env.intersections[0]].epsilon, 4),
        })

        if not quiet and (episode % log_every == 0 or episode == 1):
            elapsed = time.time() - start
            print(f"[{mode:9s}] episode {episode:5d}/{episodes} "
                  f"| file moy. {avg_queue:6.2f} | sorties {episode_exits:4d} "
                  f"| epsilon {agents[env.intersections[0]].epsilon:.3f} "
                  f"| {elapsed:5.1f}s")

    # Sauvegarde des Q-tables
    qtable_dir = os.path.join(output_dir, "qtables")
    os.makedirs(qtable_dir, exist_ok=True)
    for i in env.intersections:
        r, c = i
        path = os.path.join(qtable_dir, f"{mode}_agent_{r}_{c}.pkl")
        agents[i].save(path)

    # Sauvegarde du log d'entrainement
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "avg_queue", "total_exits", "epsilon"])
        writer.writeheader()
        writer.writerows(log_rows)

    meta = {
        "mode": mode,
        "episodes": episodes,
        "grid_size": grid_size,
        "seed": seed,
        "training_time_seconds": round(time.time() - start, 1),
        "final_avg_queue_last_100": round(
            sum(r["avg_queue"] for r in log_rows[-100:]) / min(100, len(log_rows)), 3
        ),
    }
    with open(os.path.join(output_dir, f"train_meta_{mode}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if not quiet:
        print(f"-> Q-tables sauvegardees dans {qtable_dir}")
        print(f"-> log d'entrainement dans {log_path}")
        print(f"-> resume : {meta}")

    return agents, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraine des agents Q-Learning pour la regulation de trafic.")
    parser.add_argument("--mode", choices=["local", "coordonne"], required=True)
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="artifacts")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    train(
        mode=args.mode,
        episodes=args.episodes,
        grid_size=args.grid_size,
        seed=args.seed,
        output_dir=args.output,
        quiet=args.quiet,
    )
