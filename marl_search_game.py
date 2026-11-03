from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

GRID_SIZE = 8
MAX_STEPS = 50
GOAL = (0, 0)
RUNNER_START = (7, 7)
HUNTER_START = (2, 2)
WALLS = {(1, 3), (2, 3), (3, 3), (4, 5), (5, 5), (6, 2)}
ACTIONS = ("north", "south", "west", "east", "wait")


def smooth(data: np.ndarray, window: int = 7) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64)
    if len(values) == 0:
        return values
    sums = np.cumsum(np.insert(values, 0, 0.0))
    starts = np.maximum(0, np.arange(len(values)) - window + 1)
    counts = np.arange(1, len(values) + 1) - starts
    return (sums[1:] - sums[starts]) / counts


class SearchGameEnv:
    def __init__(self, grid_size: int = GRID_SIZE, max_steps: int = MAX_STEPS) -> None:
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.move_lookup = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))
        self.reset()

    def reset(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        self.runner_pos = RUNNER_START
        self.hunter_pos = HUNTER_START
        self.step_count = 0
        self.done = False
        return self.runner_pos, self.hunter_pos

    def move(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        d_row, d_col = self.move_lookup[action]
        row = max(0, min(self.grid_size - 1, pos[0] + d_row))
        col = max(0, min(self.grid_size - 1, pos[1] + d_col))
        candidate = (row, col)
        return pos if candidate in WALLS else candidate

    def hunter_observation(self, last_seen: Tuple[int, int]) -> Tuple[int, int]:
        distance = max(
            abs(self.runner_pos[0] - self.hunter_pos[0]),
            abs(self.runner_pos[1] - self.hunter_pos[1]),
        )
        return self.runner_pos if distance <= 3 else last_seen

    def step(self, runner_action: int, hunter_action: int):
        old_runner_pos = self.runner_pos
        old_hunter_pos = self.hunter_pos
        self.runner_pos = self.move(self.runner_pos, runner_action)
        self.hunter_pos = self.move(self.hunter_pos, hunter_action)
        self.step_count += 1

        crossed_paths = (
            self.runner_pos == old_hunter_pos
            and self.hunter_pos == old_runner_pos
        )
        intercepted = self.runner_pos == self.hunter_pos or crossed_paths
        recovered = self.runner_pos == GOAL
        timeout = self.step_count >= self.max_steps

        if intercepted:
            runner_reward, hunter_reward = -12.0, 12.0
            outcome = "intercepted"
            self.done = True
        elif recovered:
            runner_reward, hunter_reward = 12.0, -12.0
            outcome = "recovered"
            self.done = True
        elif timeout:
            runner_reward, hunter_reward = -3.0, 1.0
            outcome = "timeout"
            self.done = True
        else:
            goal_distance = abs(self.runner_pos[0] - GOAL[0]) + abs(self.runner_pos[1] - GOAL[1])
            hunter_distance = abs(self.runner_pos[0] - self.hunter_pos[0]) + abs(self.runner_pos[1] - self.hunter_pos[1])
            runner_reward = -0.10 - 0.025 * goal_distance
            hunter_reward = -0.08 - 0.020 * hunter_distance
            outcome = "active"

        return (self.runner_pos, self.hunter_pos), (runner_reward, hunter_reward), self.done, outcome


class QAgent:
    def __init__(
        self,
        name: str,
        alpha: float = 0.12,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.04,
        epsilon_decay: float = 0.996,
    ) -> None:
        self.name = name
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: np.zeros(len(ACTIONS), dtype=np.float64))

    def get_action(self, state, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randrange(len(ACTIONS))
        q_values = self.q_table[state]
        return int(np.random.choice(np.flatnonzero(q_values == q_values.max())))

    def update(self, state, action: int, reward: float, next_state, done: bool) -> None:
        current_q = self.q_table[state][action]
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def make_states(env: SearchGameEnv, last_seen: Tuple[int, int]):
    runner_state = (env.runner_pos, env.hunter_pos)
    hunter_state = (env.hunter_pos, last_seen)
    return runner_state, hunter_state


def evaluate(env: SearchGameEnv, runner: QAgent, hunter: QAgent, episodes: int = 60) -> float:
    hunter_wins = 0
    for _ in range(episodes):
        env.reset()
        last_seen = env.hunter_observation((-1, -1))
        runner_state, hunter_state = make_states(env, last_seen)

        while not env.done:
            runner_action = runner.get_action(runner_state, explore=False)
            hunter_action = hunter.get_action(hunter_state, explore=False)
            _, _, done, outcome = env.step(runner_action, hunter_action)
            last_seen = env.hunter_observation(last_seen)
            runner_state, hunter_state = make_states(env, last_seen)
            if done and outcome == "intercepted":
                hunter_wins += 1

    return 100.0 * hunter_wins / episodes


def train(episodes: int = 3500, log_interval: int = 50, seed: int = 7):
    random.seed(seed)
    np.random.seed(seed)

    env = SearchGameEnv()
    runner = QAgent("artifact runner")
    hunter = QAgent("signal hunter")

    runner_success = []
    hunter_success = []
    timeout_rate = []
    runner_rewards = []
    hunter_rewards = []
    evaluation_episodes = []
    evaluation_hunter_success = []
    episode_lengths = []
    early_actions = np.zeros(len(ACTIONS), dtype=np.int64)
    late_actions = np.zeros(len(ACTIONS), dtype=np.int64)
    window_outcomes = {"recovered": 0, "intercepted": 0, "timeout": 0}
    window_runner_reward = 0.0
    window_hunter_reward = 0.0

    for episode in range(episodes):
        env.reset()
        last_seen = env.hunter_observation((-1, -1))
        runner_state, hunter_state = make_states(env, last_seen)
        episode_runner_reward = 0.0
        episode_hunter_reward = 0.0

        while not env.done:
            runner_action = runner.get_action(runner_state)
            hunter_action = hunter.get_action(hunter_state)

            if episode < 500:
                early_actions[runner_action] += 1
            elif episode >= episodes - 500:
                late_actions[runner_action] += 1

            _, (runner_reward, hunter_reward), done, outcome = env.step(runner_action, hunter_action)
            last_seen = env.hunter_observation(last_seen)
            next_runner_state, next_hunter_state = make_states(env, last_seen)

            runner.update(runner_state, runner_action, runner_reward, next_runner_state, done)
            hunter.update(hunter_state, hunter_action, hunter_reward, next_hunter_state, done)
            runner_state, hunter_state = next_runner_state, next_hunter_state
            episode_runner_reward += runner_reward
            episode_hunter_reward += hunter_reward

        window_outcomes[outcome] += 1
        window_runner_reward += episode_runner_reward
        window_hunter_reward += episode_hunter_reward
        episode_lengths.append(env.step_count)
        runner.decay_epsilon()
        hunter.decay_epsilon()

        if (episode + 1) % log_interval == 0:
            runner_success.append(100.0 * window_outcomes["recovered"] / log_interval)
            hunter_success.append(100.0 * window_outcomes["intercepted"] / log_interval)
            timeout_rate.append(100.0 * window_outcomes["timeout"] / log_interval)
            runner_rewards.append(window_runner_reward / log_interval)
            hunter_rewards.append(window_hunter_reward / log_interval)
            window_outcomes = {"recovered": 0, "intercepted": 0, "timeout": 0}
            window_runner_reward = 0.0
            window_hunter_reward = 0.0

        if (episode + 1) % 500 == 0:
            evaluation_episodes.append(episode + 1)
            evaluation_hunter_success.append(evaluate(env, runner, hunter))

    outcome_mass = np.asarray(runner_success) + np.asarray(hunter_success) + np.asarray(timeout_rate)
    assert np.allclose(outcome_mass, 100.0), "terminal outcome probabilities must sum to 100%"
    assert env.move(GOAL, 0) == GOAL, "grid boundary transition is invalid"
    assert GOAL not in WALLS and RUNNER_START not in WALLS and HUNTER_START not in WALLS, "arena geometry is invalid"

    return {
        "env": env,
        "runner": runner,
        "hunter": hunter,
        "runner_success": np.asarray(runner_success),
        "hunter_success": np.asarray(hunter_success),
        "timeout_rate": np.asarray(timeout_rate),
        "runner_rewards": np.asarray(runner_rewards),
        "hunter_rewards": np.asarray(hunter_rewards),
        "evaluation_episodes": np.asarray(evaluation_episodes),
        "evaluation_hunter_success": np.asarray(evaluation_hunter_success),
        "episode_lengths": np.asarray(episode_lengths),
        "early_actions": early_actions,
        "late_actions": late_actions,
        "log_interval": log_interval,
        "episodes": episodes,
    }


def greedy_footprints(env: SearchGameEnv, runner: QAgent, hunter: QAgent, episodes: int = 180):
    runner_footprint = np.zeros((env.grid_size, env.grid_size), dtype=np.float64)
    hunter_footprint = np.zeros((env.grid_size, env.grid_size), dtype=np.float64)

    for _ in range(episodes):
        env.reset()
        last_seen = env.hunter_observation((-1, -1))
        runner_state, hunter_state = make_states(env, last_seen)

        while not env.done:
            runner_footprint[env.runner_pos] += 1
            hunter_footprint[env.hunter_pos] += 1
            runner_action = runner.get_action(runner_state, explore=False)
            hunter_action = hunter.get_action(hunter_state, explore=False)
            env.step(runner_action, hunter_action)
            last_seen = env.hunter_observation(last_seen)
            runner_state, hunter_state = make_states(env, last_seen)

    return runner_footprint, hunter_footprint


def style_axis(ax) -> None:
    ax.set_facecolor("#151A26")
    ax.tick_params(colors="#CAD3E3", labelsize=9)
    ax.xaxis.label.set_color("#CAD3E3")
    ax.yaxis.label.set_color("#CAD3E3")
    for spine in ax.spines.values():
        spine.set_color("#3A4255")


def plot_search_game_analysis(metrics: Dict[str, object], output: Path) -> None:
    c_bg = "#10131D"
    c_panel = "#151A26"
    c_text = "#CAD3E3"
    c_muted = "#75809A"
    c_mint = "#7BE0C3"
    c_pink = "#FF6B9A"
    c_purple = "#BE9CFF"
    c_gold = "#FFD166"

    runner_footprint, hunter_footprint = greedy_footprints(
        metrics["env"], metrics["runner"], metrics["hunter"]
    )
    env = metrics["env"]
    episodes_axis = np.arange(
        metrics["log_interval"], metrics["episodes"] + 1, metrics["log_interval"]
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "mathtext.default": "regular",
    })
    fig = plt.figure(figsize=(18.5, 6.3), facecolor=c_bg)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.16, 1.0, 0.96], wspace=0.30)
    ax1 = fig.add_subplot(grid[0])
    ax2 = fig.add_subplot(grid[1])
    ax3 = fig.add_subplot(grid[2])

    for axis in (ax1, ax2, ax3):
        style_axis(axis)

    ax1.plot(episodes_axis, smooth(metrics["runner_success"]), color=c_mint, lw=2.5, label="runner recovered artifact")
    ax1.plot(episodes_axis, smooth(metrics["hunter_success"]), color=c_pink, lw=2.5, label="hunter interception")
    ax1.plot(episodes_axis, smooth(metrics["timeout_rate"]), color=c_muted, lw=1.6, ls=":", label="containment timeout")
    ax1.plot(
        metrics["evaluation_episodes"],
        metrics["evaluation_hunter_success"],
        color=c_purple,
        lw=1.8,
        ls="--",
        marker="o",
        ms=4,
        label="greedy hunter eval",
    )
    ax1.set_title("a. self-play pursuit / evasion", color=c_text, fontsize=11.5, pad=12)
    ax1.set_xlabel("training episodes")
    ax1.set_ylabel("terminal frequency (%)")
    ax1.set_ylim(-2, 102)
    ax1.grid(True, ls="--", alpha=0.28, color="#62708C")
    legend = ax1.legend(frameon=True, facecolor=c_panel, edgecolor="#3A4255", fontsize=8.0, loc="upper left")
    for text in legend.get_texts():
        text.set_color(c_text)

    base = np.tile(np.asarray(mcolors.to_rgb(c_bg)), (env.grid_size, env.grid_size, 1))
    runner_intensity = np.log1p(runner_footprint)
    hunter_intensity = np.log1p(hunter_footprint)
    runner_intensity /= max(float(runner_intensity.max()), 1e-12)
    hunter_intensity /= max(float(hunter_intensity.max()), 1e-12)
    composite = base * (1.0 - 0.64 * runner_intensity[..., None]) + np.asarray(mcolors.to_rgb(c_mint)) * 0.64 * runner_intensity[..., None]
    composite = composite * (1.0 - 0.64 * hunter_intensity[..., None]) + np.asarray(mcolors.to_rgb(c_pink)) * 0.64 * hunter_intensity[..., None]
    ax2.imshow(composite, origin="upper", interpolation="nearest")

    for row, col in WALLS:
        ax2.add_patch(patches.Rectangle((col - 0.5, row - 0.5), 1.0, 1.0, facecolor="#30384B", edgecolor="#56617B", lw=1.0, zorder=3))
        ax2.add_patch(patches.Polygon(
            [(col - 0.5, row - 0.5), (col + 0.5, row - 0.5), (col + 0.34, row - 0.68), (col - 0.66, row - 0.68)],
            closed=True,
            facecolor="#66738F",
            edgecolor="none",
            alpha=0.95,
            zorder=4,
        ))

    ax2.scatter(GOAL[1], GOAL[0], marker="*", s=185, color=c_gold, edgecolor="#FFE7A5", linewidth=0.8, zorder=6)
    ax2.scatter(RUNNER_START[1], RUNNER_START[0], marker="o", s=60, color=c_mint, edgecolor=c_bg, linewidth=1.0, zorder=6)
    ax2.scatter(HUNTER_START[1], HUNTER_START[0], marker="X", s=62, color=c_pink, edgecolor=c_bg, linewidth=1.0, zorder=6)
    ax2.set_title("b. learned search density / voxel arena", color=c_text, fontsize=11.5, pad=12)
    ax2.set_xlabel("arena column")
    ax2.set_ylabel("arena row")
    ax2.set_xticks(np.arange(env.grid_size))
    ax2.set_yticks(np.arange(env.grid_size))
    ax2.set_xticks(np.arange(-0.5, env.grid_size, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, env.grid_size, 1), minor=True)
    ax2.grid(which="minor", color="#4A5369", alpha=0.6, lw=0.7)
    ax2.grid(False, which="major")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c_mint, markersize=6, label="runner density"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=c_pink, markersize=6, label="hunter density"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=c_gold, markersize=9, label="artifact"),
    ]
    legend = ax2.legend(handles=handles, loc="lower left", frameon=True, facecolor=c_panel, edgecolor="#3A4255", fontsize=7.7)
    for text in legend.get_texts():
        text.set_color(c_text)

    x = np.arange(len(ACTIONS))
    width = 0.34
    early_action_pct = 100.0 * metrics["early_actions"] / max(int(np.sum(metrics["early_actions"])), 1)
    late_action_pct = 100.0 * metrics["late_actions"] / max(int(np.sum(metrics["late_actions"])), 1)
    ax3.bar(x - width / 2, early_action_pct, width, color="#64708B", label="early exploration")
    ax3.bar(x + width / 2, late_action_pct, width, color=c_mint, label="late policy")
    ax3.axhline(20.0, color="#75809A", lw=1.0, ls=":", alpha=0.9)
    ax3.set_title("c. policy compression", color=c_text, fontsize=11.5, pad=12)
    ax3.set_xticks(x, ACTIONS, rotation=16, ha="right")
    ax3.set_ylabel("runner action frequency (%)")
    ax3.set_ylim(0, max(35.0, float(np.max([early_action_pct, late_action_pct])) + 7.0))
    ax3.grid(True, axis="y", ls="--", alpha=0.28, color="#62708C")
    legend = ax3.legend(frameon=True, facecolor=c_panel, edgecolor="#3A4255", fontsize=8.0, loc="upper left")
    for text in legend.get_texts():
        text.set_color(c_text)

    inset = ax3.inset_axes([0.56, 0.48, 0.39, 0.36])
    style_axis(inset)
    inset.plot(smooth(metrics["episode_lengths"], window=100), color=c_purple, lw=1.5)
    inset.axhline(MAX_STEPS, color=c_muted, lw=0.8, ls=":")
    inset.set_title("termination horizon", color=c_text, fontsize=7.5, pad=4)
    inset.tick_params(labelsize=6)
    inset.grid(True, ls="--", alpha=0.18)

    fig.suptitle("partially observed artifact search: self-play q-learning under pursuit", color="#F1F5FF", fontsize=14.0, fontweight="bold", y=0.985)
    fig.text(0.5, 0.018, "cyan = runner policy occupancy  •  magenta = hunter policy occupancy  •  raised cells = occluding walls", ha="center", color=c_muted, fontsize=8.5)
    fig.savefig(output, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    metrics = train(episodes=3500, log_interval=50, seed=7)
    output_path = Path("marl_search_game_analysis.png")
    plot_search_game_analysis(metrics, output_path)
    print(
        "checks passed: terminal outcome mass = 100.0%, "
        f"final greedy hunter interception = {metrics['evaluation_hunter_success'][-1]:.1f}%"
    )
    print(f"saved {output_path}")
