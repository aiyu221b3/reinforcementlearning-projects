import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import pandas as pd
import matplotlib.colors as mcolors

def smooth(data, window=5):
    return pd.Series(data).rolling(window, min_periods=1).mean().tolist()

class MarketEnv:
    def __init__(self, grid_size=8, m_steps=50):
        self.grid_size = grid_size
        self.m_steps = m_steps
        self.MOVE_LOOKUP = ((-1,0), (1,0), (0,-1), (0,1), (0,0))
        self.reset()

    def reset(self):
        self.trd_pos = (7, 7)
        self.mm_pos = (2, 2)
        self.step_count = 0
        self.done = False
        return self.trd_pos, self.mm_pos

    def step(self, trd_action, mm_action):
        self.trd_pos = self.move(self.trd_pos, trd_action)
        self.mm_pos = self.move(self.mm_pos, mm_action)
        self.step_count += 1

        detected = self.trd_pos == self.mm_pos
        accumulated = self.trd_pos == (0, 0)
        timeout = self.step_count >= self.m_steps

        if detected:
            trd_reward = -10
            mm_reward = 10
            self.done = True
        elif accumulated:
            trd_reward = 10
            mm_reward = -10
            self.done = True
        elif timeout:
            trd_reward = -1
            mm_reward = -1
            self.done = True
        else:
            dist_penalty = (self.trd_pos[0] + self.trd_pos[1]) * 0.01
            trd_reward = -0.1 - dist_penalty
            mm_reward = -0.1

        return (self.trd_pos, self.mm_pos), (trd_reward, mm_reward), self.done

    def move(self, pos, action):
        dx, dy = self.MOVE_LOOKUP[action]
        n_x = max(0, min(self.grid_size - 1, pos[0] + dx))
        n_y = max(0, min(self.grid_size - 1, pos[1] + dy))
        return (n_x, n_y)

class QAgent:
    def __init__(self, name, alpha=0.1, gamma=0.95, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.name = name
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: np.zeros(5))

    def get_action(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randint(0, 4)
        q_values = self.q_table[state]
        return np.random.choice(np.flatnonzero(q_values == q_values.max()))

    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        best_next_q = np.max(self.q_table[next_state])
        td_error = reward + self.gamma * best_next_q - current_q
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

def train(eps=3000, log_inte=50):
    env = MarketEnv()
    trd = QAgent("InformedTrader")
    mm = QAgent("MarketMaker")

    trd_wins = []
    mm_wins = []
    trd_rw_log = []
    mm_rw_log = []
    eval_eps = []
    eval_mm_wins = []
    early_actions = {0:0, 1:0, 2:0, 3:0, 4:0}
    late_actions = {0:0, 1:0, 2:0, 3:0, 4:0}
    ep_lengths = []

    trd_win_c = 0
    mm_win_c = 0
    trd_re_sum = 0
    mm_re_sum = 0

    for ep in range(eps):
        trd_pos, mm_pos = env.reset()
        last_known_trd = trd_pos
        trd_state = (trd_pos, mm_pos)
        mm_state = (mm_pos, last_known_trd)
        ep_trd_re = 0
        ep_mm_re = 0

        while not env.done:
            ta = trd.get_action(trd_state)
            ma = mm.get_action(mm_state)

            if ep < 500:
                early_actions[ta] += 1
            elif ep > eps - 500:
                late_actions[ta] += 1

            (new_trd_pos, new_mm_pos), (tr, mr), done = env.step(ta, ma)

            dist = max(abs(new_trd_pos[0] - new_mm_pos[0]), abs(new_trd_pos[1] - new_mm_pos[1]))
            if dist <= 3:
                last_known_trd = new_trd_pos

            next_trd_st = (new_trd_pos, new_mm_pos)
            next_mm_st = (new_mm_pos, last_known_trd)

            trd.update(trd_state, ta, tr, next_trd_st)
            mm.update(mm_state, ma, mr, next_mm_st)

            trd_state = next_trd_st
            mm_state = next_mm_st
            ep_trd_re += tr
            ep_mm_re += mr

        ep_lengths.append(env.step_count)

        if env.trd_pos == (0, 0):
            trd_win_c += 1
        elif env.trd_pos == env.mm_pos:
            mm_win_c += 1

        trd_re_sum += ep_trd_re
        mm_re_sum += ep_mm_re
        trd.decay_epsilon()
        mm.decay_epsilon()

        if (ep + 1) % log_inte == 0:
            trd_wins.append(trd_win_c / log_inte * 100)
            mm_wins.append(mm_win_c / log_inte * 100)
            trd_rw_log.append(trd_re_sum / log_inte)
            mm_rw_log.append(mm_re_sum / log_inte)
            trd_win_c = 0
            mm_win_c = 0
            trd_re_sum = 0
            mm_re_sum = 0

        if ep > 0 and ep % 500 == 0:
            eval_mm_win_c = 0
            for _ in range(50):
                e_trd_pos, e_mm_pos = env.reset()
                e_last_known = e_trd_pos
                e_trd_st = (e_trd_pos, e_mm_pos)
                e_mm_st = (e_mm_pos, e_last_known)
                while not env.done:
                    e_ta = trd.get_action(e_trd_st, explore=False)
                    e_ma = mm.get_action(e_mm_st, explore=False)
                    (n_e_trd, n_e_mm), _, _ = env.step(e_ta, e_ma)
                    e_dist = max(abs(n_e_trd[0] - n_e_mm[0]), abs(n_e_trd[1] - n_e_mm[1]))
                    if e_dist <= 3:
                        e_last_known = n_e_trd
                    e_trd_st = (n_e_trd, n_e_mm)
                    e_mm_st = (n_e_mm, e_last_known)
                if env.trd_pos == env.mm_pos:
                    eval_mm_win_c += 1
            eval_eps.append(ep)
            eval_mm_wins.append(eval_mm_win_c / 50 * 100)

    return env, trd, mm, trd_wins, mm_wins, trd_rw_log, mm_rw_log, log_inte, eps, eval_eps, eval_mm_wins, early_actions, late_actions, ep_lengths

def plot_results(trd_wins, mm_wins, trd_rw_log, mm_rw_log, log_inte, eps, eval_eps, eval_mm_wins, early_actions, late_actions):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.5,
        "grid.color": "#eaeaea",
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 1.2
    })

    COLOR_TRD = "#3a86ff"
    COLOR_MM = "#ff006e"
    COLOR_EVAL = "#8338ec"
    COLOR_EARLY = "#d9d9d9"
    
    episodes_axis = list(range(log_inte, eps + 1, log_inte))
    
    s_trd_wins = smooth(trd_wins, window=10)
    s_mm_wins = smooth(mm_wins, window=10)
    s_trd_rw = smooth(trd_rw_log, window=10)
    s_mm_rw = smooth(mm_rw_log, window=10)

    plt.figure(figsize=(8, 5))
    plt.plot(episodes_axis, s_trd_wins, label="Informed Trader", color=COLOR_TRD, linewidth=2)
    plt.plot(episodes_axis, s_mm_wins, label="Market Maker", color=COLOR_MM, linewidth=2)
    plt.title("Non-Stationarity in Co-Evolving Market Agents", pad=15)
    plt.xlabel("Training Episodes")
    plt.ylabel("Success Rate (%)")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig("marl_non_stationarity.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(episodes_axis, s_trd_rw, label="Informed Trader", color=COLOR_TRD, linewidth=2)
    plt.plot(episodes_axis, s_mm_rw, label="Market Maker", color=COLOR_MM, linewidth=2)
    plt.title("Reward Evolution Dynamics", pad=15)
    plt.xlabel("Training Episodes")
    plt.ylabel("Average Reward per Episode")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig("marl_reward_evolution.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(eval_eps, eval_mm_wins, marker='o', markersize=6, color=COLOR_EVAL, linewidth=2)
    plt.title("Self-Play Evaluation (MM Win Rate vs Frozen Trader)", pad=15)
    plt.xlabel("Training Episodes")
    plt.ylabel("Evaluation Win Rate (%)")
    plt.tight_layout()
    plt.savefig("marl_self_play.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 5))
    labels = ['Up', 'Down', 'Left', 'Right', 'Stay']
    total_early = sum(early_actions.values()) or 1
    total_late = sum(late_actions.values()) or 1
    early_vals = [(early_actions[i] / total_early) * 100 for i in range(5)]
    late_vals = [(late_actions[i] / total_late) * 100 for i in range(5)]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, early_vals, width, label='Early Training', color=COLOR_EARLY)
    plt.bar(x + width/2, late_vals, width, label='Late Training', color=COLOR_TRD)
    plt.title("Emergent Behavior: Trader Action Distribution", pad=15)
    plt.xticks(x, labels)
    plt.ylabel("Frequency (%)")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig("marl_action_distribution.pdf", format='pdf', bbox_inches='tight')
    plt.close()

def plot_advanced_metrics(env, trd, mm, ep_lengths, log_inte=50):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.5,
        "grid.color": "#eaeaea",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    COLOR_TRD = "#3a86ff"
    COLOR_MM = "#ff006e"

    plt.figure(figsize=(8, 5))
    s_lengths = smooth(ep_lengths, window=100)
    plt.plot(s_lengths, color="#2b2d42", linewidth=2, alpha=0.9)
    plt.title("Convergence of Episode Duration", pad=15)
    plt.xlabel("Training Episodes")
    plt.ylabel("Steps to Termination")
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label="Timeout Limit")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("marl_episode_duration.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    trd_footprint = np.zeros((env.grid_size, env.grid_size))
    mm_footprint = np.zeros((env.grid_size, env.grid_size))

    for _ in range(100):
        t_pos, m_pos = env.reset()
        t_state, m_state = (t_pos, m_pos), (m_pos, t_pos)
        
        while not env.done:
            trd_footprint[t_pos[1], t_pos[0]] += 1
            mm_footprint[m_pos[1], m_pos[0]] += 1
            ta = trd.get_action(t_state, explore=False)
            ma = mm.get_action(m_state, explore=False)
            (t_pos, m_pos), _, _ = env.step(ta, ma)
            t_state, m_state = (t_pos, m_pos), (m_pos, t_pos)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap_trd = mcolors.LinearSegmentedColormap.from_list("trd_map", ["white", COLOR_TRD])
    cmap_mm = mcolors.LinearSegmentedColormap.from_list("mm_map", ["white", COLOR_MM])

    im1 = axes[0].imshow(trd_footprint, cmap=cmap_trd, origin='upper')
    axes[0].set_title("Informed Trader Trajectory Density")
    axes[0].set_xticks(range(env.grid_size))
    axes[0].set_yticks(range(env.grid_size))
    axes[0].grid(False)
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(mm_footprint, cmap=cmap_mm, origin='upper')
    axes[1].set_title("Market Maker Patrol Density")
    axes[1].set_xticks(range(env.grid_size))
    axes[1].set_yticks(range(env.grid_size))
    axes[1].grid(False)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("marl_spatial_heatmap.pdf", format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Training market agents...")
    env, trd, mm, tw, mw, tr, mr, log_int, eps, e_eps, e_mw, early, late, ep_lengths = train(eps=3000, log_inte=50)

    print("Generating baseline metrics...")
    plot_results(tw, mw, tr, mr, log_int, eps, e_eps, e_mw, early, late)

    print("Generating advanced spatial and kinematic metrics...")
    plot_advanced_metrics(env, trd, mm, ep_lengths, log_inte=log_int)

    print("Downloading all publication figures...")
    from google.colab import files
    files.download("marl_non_stationarity.pdf")
    files.download("marl_reward_evolution.pdf")
    files.download("marl_self_play.pdf")
    files.download("marl_action_distribution.pdf")
    files.download("marl_episode_duration.pdf")
    files.download("marl_spatial_heatmap.pdf")
    print("Done.")