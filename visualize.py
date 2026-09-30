"""
Visualisation Pygame de la grille de trafic en temps reel.

Usage :
    python visualize.py --controller fixed
    python visualize.py --controller heuristic
    python visualize.py --controller local
    python visualize.py --controller coordonne

Controles (identiques aux autres projets de la formation) :
    1-9   : vitesse de simulation
    ESPACE: pause
    R     : reset
    ECHAP : quitter
"""

import argparse
import os
import sys

import pygame

from constants import GRID_SIZE, MAX_QUEUE, PHASE_NS
from env.grid_env import TrafficGridEnv
from agents.q_learning_agent import QLearningAgent
from baselines.fixed_time import FixedTimeController
from baselines.longest_queue import LongestQueueController

CELL_SIZE = 160
MARGIN = 60
PANEL_HEIGHT = 90
FPS = 30

COLOR_BG = (24, 26, 32)
COLOR_ROAD = (60, 63, 70)
COLOR_TEXT = (230, 230, 230)
COLOR_PANEL = (18, 20, 25)
COLOR_GREEN = (102, 187, 106)
COLOR_RED = (239, 83, 80)


def queue_color(level_ratio):
    """Vert -> jaune -> rouge selon le niveau de congestion (0..1)."""
    if level_ratio < 0.4:
        return (102, 187, 106)
    if level_ratio < 0.75:
        return (255, 202, 40)
    return (239, 83, 80)


def load_controller(kind, env, artifacts_dir):
    if kind == "fixed":
        return FixedTimeController(), False
    if kind == "heuristic":
        return LongestQueueController(), False
    if kind in ("local", "coordonne"):
        qtable_dir = os.path.join(artifacts_dir, "qtables")
        agents = {}
        for i in env.intersections:
            r, c = i
            agent = QLearningAgent(i)
            path = os.path.join(qtable_dir, f"{kind}_agent_{r}_{c}.pkl")
            if not os.path.exists(path):
                print(f"ERREUR : Q-table introuvable ({path}). "
                      f"Lance d'abord 'python train.py --mode {kind}'.")
                sys.exit(1)
            agent.load(path)
            agent.epsilon = 0.0
            agents[i] = agent
        return agents, True
    raise ValueError(f"controleur inconnu: {kind}")


def act(controller, is_qlearning, states, step_count, env):
    if is_qlearning:
        return {i: controller[i].select_action(states[i], greedy=True) for i in env.intersections}
    if isinstance(controller, FixedTimeController):
        return {i: controller.act(states[i], step_count) for i in env.intersections}
    return {i: controller.act(states[i]) for i in env.intersections}


MIN_WIDTH = 700  # assez large pour que le texte du panneau ne soit jamais coupe


def get_dimensions(grid_size):
    width = max(MIN_WIDTH, MARGIN * 2 + CELL_SIZE * grid_size)
    height = MARGIN * 2 + CELL_SIZE * grid_size + PANEL_HEIGHT
    return width, height


def render_frame(screen, env, controller_name, speed, paused, total_exits, font, font_small):
    """Dessine une image complete de l'etat courant de l'environnement.
    Reutilisee par main() (fenetre live) et par les scripts de capture d'ecran.
    """
    width, height = screen.get_size()
    screen.fill(COLOR_BG)

    grid_pixel_width = CELL_SIZE * (max(c for _r, c in env.intersections) + 1)
    x_offset = MARGIN + max(0, (width - MARGIN * 2 - grid_pixel_width) // 2)

    for (r, c) in env.intersections:
        cx = x_offset + c * CELL_SIZE + CELL_SIZE // 2
        cy = MARGIN + r * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.line(screen, COLOR_ROAD, (cx, MARGIN - 20), (cx, height - PANEL_HEIGHT + 20), 18)
        pygame.draw.line(screen, COLOR_ROAD, (x_offset - 20, cy), (x_offset + grid_pixel_width - CELL_SIZE // 2 + 40, cy), 18)

    for (r, c) in env.intersections:
        cx = x_offset + c * CELL_SIZE + CELL_SIZE // 2
        cy = MARGIN + r * CELL_SIZE + CELL_SIZE // 2
        q = env.queues[(r, c)]
        phase = env.phases[(r, c)]

        bar_len = 55
        for d, (dx, dy, horizontal) in {
            "N": (0, -1, False), "S": (0, 1, False),
            "E": (1, 0, True), "W": (-1, 0, True),
        }.items():
            ratio = min(1.0, q[d] / MAX_QUEUE)
            length = int(10 + ratio * bar_len)
            color = queue_color(ratio)
            if horizontal:
                rect = pygame.Rect(0, 0, length, 10)
                rect.centery = cy
                rect.x = cx + (25 if dx > 0 else -25 - length)
            else:
                rect = pygame.Rect(0, 0, 10, length)
                rect.centerx = cx
                rect.y = cy + (25 if dy > 0 else -25 - length)
            pygame.draw.rect(screen, color, rect, border_radius=3)
            count_label = font_small.render(str(q[d]), True, COLOR_TEXT)
            screen.blit(count_label, (rect.centerx - 6, rect.centery - 7))

        light_color = COLOR_GREEN if phase == PHASE_NS else COLOR_RED
        pygame.draw.circle(screen, light_color, (cx - 14, cy - 14), 7)
        light_color2 = COLOR_RED if phase == PHASE_NS else COLOR_GREEN
        pygame.draw.circle(screen, light_color2, (cx + 14, cy - 14), 7)
        pygame.draw.circle(screen, COLOR_BG, (cx, cy), 22)
        pygame.draw.circle(screen, COLOR_ROAD, (cx, cy), 22, 2)

    panel_rect = pygame.Rect(0, height - PANEL_HEIGHT, width, PANEL_HEIGHT)
    pygame.draw.rect(screen, COLOR_PANEL, panel_rect)
    total_queue = sum(sum(qq.values()) for qq in env.queues.values())
    status = "PAUSE" if paused else f"vitesse x{speed}"
    lines = [
        f"Controleur: {controller_name}   |   {status}   |   pas: {env.step_count}/{env.episode_length}",
        f"File totale: {total_queue}   |   Sorties: {total_exits}",
    ]
    for idx, line in enumerate(lines):
        label = font.render(line, True, COLOR_TEXT)
        screen.blit(label, (16, height - PANEL_HEIGHT + 12 + idx * 26))


def main():
    parser = argparse.ArgumentParser(description="Visualise un controleur de trafic en temps reel.")
    parser.add_argument("--controller", choices=["fixed", "heuristic", "local", "coordonne"], default="coordonne")
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--artifacts", type=str, default="artifacts")
    args = parser.parse_args()

    env = TrafficGridEnv(grid_size=args.grid_size)
    controller, is_qlearning = load_controller(args.controller, env, args.artifacts)

    pygame.init()
    pygame.display.set_caption(f"Trafic multi-agents - {args.controller}")
    width, height = get_dimensions(args.grid_size)
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    font_small = pygame.font.SysFont("consolas", 14)

    states = env.reset()
    speed = 1
    paused = False
    total_exits = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    states = env.reset()
                    total_exits = 0
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    speed = event.key - pygame.K_0

        if not paused:
            for _ in range(speed):
                actions = act(controller, is_qlearning, states, env.step_count, env)
                states, _rewards, done, info = env.step(actions)
                total_exits += info["exits_this_step"]
                if done:
                    states = env.reset()
                    total_exits = 0

        render_frame(screen, env, args.controller, speed, paused, total_exits, font, font_small)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pygame.quit()
        print("\nFerme (Ctrl+C).")
