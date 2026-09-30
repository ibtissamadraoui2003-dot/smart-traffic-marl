"""
Compare les 4 controleurs de trafic sur des episodes d'evaluation identiques :

    1. Feux fixes               (baseline "aucune IA")
    2. File la plus longue       (baseline heuristique)
    3. Q-Learning local          (agents egoistes)
    4. Q-Learning coordonne      (agents qui tiennent compte de leurs voisins)

Produit results/comparison.csv, results/comparison_queue.png,
results/comparison_throughput.png et results/learning_curves.png.

Usage :
    python evaluate.py --episodes 30
"""

import argparse
import csv
import os
import statistics as stats

import matplotlib
matplotlib.use("Agg")  # pas d'affichage requis, on sauvegarde direct en PNG
import matplotlib.pyplot as plt

from constants import GRID_SIZE
from env.grid_env import TrafficGridEnv
from agents.q_learning_agent import QLearningAgent
from baselines.fixed_time import FixedTimeController
from baselines.longest_queue import LongestQueueController


def run_episode_fixed_or_heuristic(env, controller, seed):
    env.rng.seed(seed)
    states = env.reset()
    queue_hist = []
    max_queue = 0
    total_exits = 0
    for step in range(env.episode_length):
        if isinstance(controller, FixedTimeController):
            actions = {i: controller.act(states[i], step) for i in env.intersections}
        else:
            actions = {i: controller.act(states[i]) for i in env.intersections}
        states, _rewards, done, info = env.step(actions)
        queue_hist.append(info["total_queue"])
        total_exits += info["exits_this_step"]
        max_queue = max(max_queue, env.get_max_queue_snapshot())
        if done:
            break
    return {
        "avg_queue": sum(queue_hist) / len(queue_hist),
        "total_exits": total_exits,
        "max_queue": max_queue,
    }


def run_episode_qlearning(env, agents, seed):
    env.rng.seed(seed)
    states = env.reset()
    queue_hist = []
    max_queue = 0
    total_exits = 0
    for _step in range(env.episode_length):
        actions = {i: agents[i].select_action(states[i], greedy=True) for i in env.intersections}
        states, _rewards, done, info = env.step(actions)
        queue_hist.append(info["total_queue"])
        total_exits += info["exits_this_step"]
        max_queue = max(max_queue, env.get_max_queue_snapshot())
        if done:
            break
    return {
        "avg_queue": sum(queue_hist) / len(queue_hist),
        "total_exits": total_exits,
        "max_queue": max_queue,
    }


def load_agents(env, mode, qtable_dir):
    agents = {}
    for i in env.intersections:
        r, c = i
        agent = QLearningAgent(i)
        path = os.path.join(qtable_dir, f"{mode}_agent_{r}_{c}.pkl")
        agent.load(path)
        agent.epsilon = 0.0  # evaluation greedy pure
        agents[i] = agent
    return agents


def summarize(name, episode_results):
    avg_queues = [r["avg_queue"] for r in episode_results]
    throughputs = [r["total_exits"] for r in episode_results]
    max_queues = [r["max_queue"] for r in episode_results]
    return {
        "controller": name,
        "avg_queue_mean": round(stats.mean(avg_queues), 2),
        "avg_queue_std": round(stats.pstdev(avg_queues), 2),
        "throughput_mean": round(stats.mean(throughputs), 1),
        "throughput_std": round(stats.pstdev(throughputs), 1),
        "max_queue_mean": round(stats.mean(max_queues), 2),
    }


def evaluate(episodes, grid_size, artifacts_dir, output_dir, seed_base=1000):
    os.makedirs(output_dir, exist_ok=True)
    env = TrafficGridEnv(grid_size=grid_size, seed=seed_base)
    qtable_dir = os.path.join(artifacts_dir, "qtables")

    fixed_ctrl = FixedTimeController()
    heuristic_ctrl = LongestQueueController()
    local_agents = load_agents(env, "local", qtable_dir)
    coord_agents = load_agents(env, "coordonne", qtable_dir)

    controllers = [
        ("Feux fixes", "fixed"),
        ("File la plus longue", "heuristic"),
        ("Q-Learning local", "local"),
        ("Q-Learning coordonne", "coordonne"),
    ]

    all_results = {}
    for label, key in controllers:
        episode_results = []
        for ep in range(episodes):
            seed = seed_base + ep
            if key == "fixed":
                res = run_episode_fixed_or_heuristic(env, fixed_ctrl, seed)
            elif key == "heuristic":
                res = run_episode_fixed_or_heuristic(env, heuristic_ctrl, seed)
            elif key == "local":
                res = run_episode_qlearning(env, local_agents, seed)
            else:
                res = run_episode_qlearning(env, coord_agents, seed)
            episode_results.append(res)
        all_results[key] = summarize(label, episode_results)
        print(f"{label:22s} | file moy. {all_results[key]['avg_queue_mean']:6.2f} "
              f"(+/-{all_results[key]['avg_queue_std']:5.2f}) "
              f"| debit {all_results[key]['throughput_mean']:6.1f} "
              f"| pic max {all_results[key]['max_queue_mean']:5.2f}")

    # CSV
    csv_path = os.path.join(output_dir, "comparison.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["controller", "avg_queue_mean", "avg_queue_std",
                      "throughput_mean", "throughput_std", "max_queue_mean"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _label, key in controllers:
            writer.writerow(all_results[key])

    _plot_comparison(all_results, controllers, output_dir)
    _plot_learning_curves(artifacts_dir, output_dir)

    baseline_avg = all_results["fixed"]["avg_queue_mean"]
    coord_avg = all_results["coordonne"]["avg_queue_mean"]
    improvement = 100 * (baseline_avg - coord_avg) / baseline_avg
    print(f"\n>> Amelioration Q-Learning coordonne vs feux fixes : {improvement:.1f} % de file d'attente en moins")

    return all_results


def _plot_comparison(all_results, controllers, output_dir):
    labels = [label for label, _ in controllers]
    keys = [key for _, key in controllers]

    # Graphique 1 : file d'attente moyenne (proxy du temps d'attente)
    means = [all_results[k]["avg_queue_mean"] for k in keys]
    stds = [all_results[k]["avg_queue_std"] for k in keys]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, means, yerr=stds, capsize=5,
                    color=["#9e9e9e", "#78909c", "#42a5f5", "#66bb6a"])
    plt.ylabel("File d'attente moyenne (proxy du temps d'attente)")
    plt.title("Comparaison des controleurs de trafic")
    plt.xticks(rotation=15, ha="right")
    for bar, m in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(means) * 0.02,
                  f"{m:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_queue.png"), dpi=150)
    plt.close()

    # Graphique 2 : debit (vehicules sortis de la grille)
    throughputs = [all_results[k]["throughput_mean"] for k in keys]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, throughputs, color=["#9e9e9e", "#78909c", "#42a5f5", "#66bb6a"])
    plt.ylabel("Vehicules ecoules par episode (debit)")
    plt.title("Debit de trafic par controleur")
    plt.xticks(rotation=15, ha="right")
    for bar, t in zip(bars, throughputs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(throughputs) * 0.02,
                  f"{t:.0f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_throughput.png"), dpi=150)
    plt.close()


def _rolling_mean(values, window=50):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo:i + 1]) / (i - lo + 1))
    return out


def _plot_learning_curves(artifacts_dir, output_dir):
    plt.figure(figsize=(8, 5))
    for mode, color, label in [
        ("local", "#42a5f5", "Q-Learning local (egoiste)"),
        ("coordonne", "#66bb6a", "Q-Learning coordonne"),
    ]:
        log_path = os.path.join(artifacts_dir, f"train_log_{mode}.csv")
        if not os.path.exists(log_path):
            continue
        episodes, avg_queues = [], []
        with open(log_path) as f:
            for row in csv.DictReader(f):
                episodes.append(int(row["episode"]))
                avg_queues.append(float(row["avg_queue"]))
        smoothed = _rolling_mean(avg_queues, window=50)
        plt.plot(episodes, smoothed, color=color, label=label, linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("File d'attente moyenne (moyenne glissante sur 50 episodes)")
    plt.title("Courbe d'apprentissage : local vs coordonne")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curves.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalue et compare les controleurs de trafic.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--artifacts", type=str, default="artifacts")
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    evaluate(
        episodes=args.episodes,
        grid_size=args.grid_size,
        artifacts_dir=args.artifacts,
        output_dir=args.output,
    )
