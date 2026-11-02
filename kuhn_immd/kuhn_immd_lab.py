from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


CARDS = ("J", "Q", "K")
P0_INFOSETS = ("J | ∅", "Q | ∅", "K | ∅", "J | PB", "Q | PB", "K | PB")
P1_INFOSETS = ("J | P", "Q | P", "K | P", "J | B", "Q | B", "K | B")


@dataclass(frozen=True)
class MMDTrace:
    row: np.ndarray
    col: np.ndarray
    regularized_gap: np.ndarray
    nash_conv: np.ndarray


@dataclass(frozen=True)
class IMMDTrace:
    row: np.ndarray
    col: np.ndarray
    nash_conv: np.ndarray
    inner_residual: np.ndarray


@dataclass(frozen=True)
class CFRTrace:
    last_nash_conv: np.ndarray
    average_nash_conv: np.ndarray
    last_policy_0: np.ndarray
    last_policy_1: np.ndarray
    average_policy_0: np.ndarray
    average_policy_1: np.ndarray


def pure_strategies() -> np.ndarray:
    return np.asarray(list(product((0, 1), repeat=6)), dtype=np.int8)


def showdown(c0: int, c1: int, stake: float) -> float:
    return stake if c0 > c1 else -stake


def pure_payoff(s0: np.ndarray, s1: np.ndarray, c0: int, c1: int) -> float:
    if s0[c0] == 0:
        if s1[c1] == 0:
            return showdown(c0, c1, 1.0)
        return -1.0 if s0[3 + c0] == 0 else showdown(c0, c1, 2.0)
    return 1.0 if s1[3 + c1] == 0 else showdown(c0, c1, 2.0)


def build_payoff_matrix(pure: np.ndarray) -> np.ndarray:
    mat = np.empty((len(pure), len(pure)), dtype=np.float64)
    deals = tuple((c0, c1) for c0 in range(3) for c1 in range(3) if c0 != c1)
    for i, s0 in enumerate(pure):
        for j, s1 in enumerate(pure):
            mat[i, j] = np.mean([pure_payoff(s0, s1, c0, c1) for c0, c1 in deals])
    return mat


def normalize(vals: np.ndarray) -> np.ndarray:
    total = float(np.sum(vals))
    if total <= 0.0:
        return np.full_like(vals, 1.0 / len(vals), dtype=np.float64)
    return vals / total


def policy_from_regrets(regrets: np.ndarray) -> np.ndarray:
    pos = np.maximum(regrets, 0.0)
    out = np.empty_like(pos)
    for i, row in enumerate(pos):
        out[i] = normalize(row)
    return out


def normal_form_from_policy(policy: np.ndarray, pure: np.ndarray) -> np.ndarray:
    probs = np.ones(len(pure), dtype=np.float64)
    for idx in range(6):
        probs *= policy[idx, pure[:, idx]]
    return normalize(probs)


def conditional_action_probs(
    pure: np.ndarray,
    mix: np.ndarray,
    idx: int,
    cond_idx: int | None = None,
    cond_action: int | None = None,
) -> np.ndarray:
    mask = np.ones(len(pure), dtype=bool)
    if cond_idx is not None and cond_action is not None:
        mask &= pure[:, cond_idx] == cond_action
    mass = float(np.sum(mix[mask]))
    if mass <= 1e-15:
        return np.array([0.5, 0.5], dtype=np.float64)
    return np.array(
        [np.sum(mix[mask & (pure[:, idx] == act)]) / mass for act in range(2)],
        dtype=np.float64,
    )


def behavioral_from_normal_form(pure: np.ndarray, row: np.ndarray, col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p0 = np.empty((6, 2), dtype=np.float64)
    p1 = np.empty((6, 2), dtype=np.float64)
    for card in range(3):
        p0[card] = conditional_action_probs(pure, row, card)
        p0[3 + card] = conditional_action_probs(pure, row, 3 + card, card, 0)
        p1[card] = conditional_action_probs(pure, col, card)
        p1[3 + card] = conditional_action_probs(pure, col, 3 + card)
    return p0, p1


def behavioral_value(p0: np.ndarray, p1: np.ndarray) -> float:
    total = 0.0
    for c0 in range(3):
        for c1 in range(3):
            if c0 == c1:
                continue
            checked = p0[c0, 0] * (
                p1[c1, 0] * showdown(c0, c1, 1.0)
                + p1[c1, 1]
                * (p0[3 + c0, 0] * -1.0 + p0[3 + c0, 1] * showdown(c0, c1, 2.0))
            )
            bet = p0[c0, 1] * (p1[3 + c1, 0] * 1.0 + p1[3 + c1, 1] * showdown(c0, c1, 2.0))
            total += (checked + bet) / 6.0
    return total


def nash_conv(mat: np.ndarray, row: np.ndarray, col: np.ndarray) -> float:
    return float(np.max(mat @ col) - np.min(row @ mat))


def kl(probs: np.ndarray, ref: np.ndarray) -> float:
    p = np.clip(probs, 1e-300, 1.0)
    r = np.clip(ref, 1e-300, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(r))))


def logsumexp(vals: np.ndarray) -> float:
    shift = float(np.max(vals))
    return shift + float(np.log(np.sum(np.exp(vals - shift))))


def regularized_gap(
    mat: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    row_ref: np.ndarray,
    col_ref: np.ndarray,
    tau: float,
) -> float:
    row_term = tau * logsumexp(np.log(row_ref) + (mat @ col) / tau)
    col_term = tau * logsumexp(np.log(col_ref) - (mat.T @ row) / tau)
    return float(row_term + col_term + tau * (kl(row, row_ref) + kl(col, col_ref)))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    return np.exp(shifted) / np.sum(np.exp(shifted))


def fixed_mmd(
    mat: np.ndarray,
    row_ref: np.ndarray,
    col_ref: np.ndarray,
    tau: float,
    eta: float,
    steps: int,
) -> MMDTrace:
    row = row_ref.copy()
    col = col_ref.copy()
    reg_hist = np.empty(steps + 1, dtype=np.float64)
    nc_hist = np.empty(steps + 1, dtype=np.float64)
    den = 1.0 + eta * tau
    log_row_ref = np.log(row_ref)
    log_col_ref = np.log(col_ref)
    for step in range(steps + 1):
        reg_hist[step] = regularized_gap(mat, row, col, row_ref, col_ref, tau)
        nc_hist[step] = nash_conv(mat, row, col)
        if step == steps:
            break
        old_row = row
        old_col = col
        row = softmax((np.log(old_row) + eta * tau * log_row_ref + eta * (mat @ old_col)) / den)
        col = softmax((np.log(old_col) + eta * tau * log_col_ref - eta * (mat.T @ old_row)) / den)
    return MMDTrace(row, col, reg_hist, nc_hist)


def iterative_mmd(
    mat: np.ndarray,
    tau: float,
    eta: float,
    rounds: int,
    inner_steps: int,
) -> IMMDTrace:
    size = mat.shape[0]
    row = np.full(size, 1.0 / size, dtype=np.float64)
    col = np.full(size, 1.0 / size, dtype=np.float64)
    nc_hist = np.empty(rounds + 1, dtype=np.float64)
    residuals = np.empty(rounds, dtype=np.float64)
    nc_hist[0] = nash_conv(mat, row, col)
    for rnd in range(rounds):
        inner = fixed_mmd(mat, row, col, tau=tau, eta=eta, steps=inner_steps)
        row = inner.row
        col = inner.col
        nc_hist[rnd + 1] = nash_conv(mat, row, col)
        residuals[rnd] = inner.regularized_gap[-1]
    return IMMDTrace(row, col, nc_hist, residuals)


def terminal_value(hist: str, c0: int, c1: int) -> float | None:
    if hist == "pp":
        return showdown(c0, c1, 1.0)
    if hist == "pbf":
        return -1.0
    if hist == "pbc":
        return showdown(c0, c1, 2.0)
    if hist == "bf":
        return 1.0
    if hist == "bc":
        return showdown(c0, c1, 2.0)
    return None


def next_history(hist: str, action: int) -> str:
    if hist == "":
        return "p" if action == 0 else "b"
    if hist == "p":
        return "pp" if action == 0 else "pb"
    if hist == "pb":
        return "pbf" if action == 0 else "pbc"
    if hist == "b":
        return "bf" if action == 0 else "bc"
    raise ValueError(f"unexpected non-terminal history: {hist}")


def player_to_act(hist: str) -> int:
    if hist in ("", "pb"):
        return 0
    if hist in ("p", "b"):
        return 1
    raise ValueError(f"terminal history has no player: {hist}")


def infoset_index(player: int, card: int, hist: str) -> int:
    if player == 0:
        return card if hist == "" else 3 + card
    return card if hist == "p" else 3 + card


def run_cfr_plus(mat: np.ndarray, iters: int, delay: int = 20) -> CFRTrace:
    regrets = np.zeros((12, 2), dtype=np.float64)
    strategy_sum = np.zeros((12, 2), dtype=np.float64)
    last_hist = np.empty(iters, dtype=np.float64)
    avg_hist = np.empty(iters, dtype=np.float64)

    for turn in range(1, iters + 1):
        policy = policy_from_regrets(regrets)
        regret_delta = np.zeros_like(regrets)
        average_delta = np.zeros_like(strategy_sum)

        def traverse(c0: int, c1: int, hist: str, r0: float, r1: float) -> float:
            value = terminal_value(hist, c0, c1)
            if value is not None:
                return value
            player = player_to_act(hist)
            card = c0 if player == 0 else c1
            idx = infoset_index(player, card, hist)
            strat = policy[idx if player == 0 else 6 + idx]
            children = np.array(
                [
                    traverse(c0, c1, next_history(hist, act), r0 * strat[act], r1)
                    if player == 0
                    else traverse(c0, c1, next_history(hist, act), r0, r1 * strat[act])
                    for act in range(2)
                ],
                dtype=np.float64,
            )
            node = float(np.dot(strat, children))
            chance = 1.0 / 6.0
            if player == 0:
                regret_delta[idx] += chance * r1 * (children - node)
                average_delta[idx] += chance * r0 * strat
            else:
                regret_delta[6 + idx] += chance * r0 * (node - children)
                average_delta[6 + idx] += chance * r1 * strat
            return node

        for c0 in range(3):
            for c1 in range(3):
                if c0 != c1:
                    traverse(c0, c1, "", 1.0, 1.0)

        regrets = np.maximum(regrets + regret_delta, 0.0)
        weight = float(max(0, turn - delay))
        strategy_sum += weight * average_delta
        last_policy = policy_from_regrets(regrets)
        average_policy = np.vstack(
            [normalize(strategy_sum[i]) if np.sum(strategy_sum[i]) > 0.0 else np.array([0.5, 0.5]) for i in range(12)]
        )
        last_hist[turn - 1] = nash_conv(
            mat,
            normal_form_from_policy(last_policy[:6], PURE),
            normal_form_from_policy(last_policy[6:], PURE),
        )
        avg_hist[turn - 1] = nash_conv(
            mat,
            normal_form_from_policy(average_policy[:6], PURE),
            normal_form_from_policy(average_policy[6:], PURE),
        )

    return CFRTrace(
        last_hist,
        avg_hist,
        last_policy[:6],
        last_policy[6:],
        average_policy[:6],
        average_policy[6:],
    )


def analytic_nash(alpha: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= alpha <= 1.0 / 3.0:
        raise ValueError("alpha must belong to [0, 1/3]")
    p0 = np.array(
        [
            [1.0 - alpha, alpha],
            [1.0, 0.0],
            [1.0 - 3.0 * alpha, 3.0 * alpha],
            [1.0, 0.0],
            [2.0 / 3.0 - alpha, 1.0 / 3.0 + alpha],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    p1 = np.array(
        [
            [2.0 / 3.0, 1.0 / 3.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0 / 3.0, 1.0 / 3.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return p0, p1


def behavioral_bands() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    low0 = np.array([0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 1.0])
    high0 = np.array([1.0 / 3.0, 0.0, 1.0, 0.0, 2.0 / 3.0, 1.0])
    low1 = np.array([1.0 / 3.0, 0.0, 1.0, 0.0, 1.0 / 3.0, 1.0])
    high1 = low1.copy()
    return low0, high0, low1, high1


def run_checks(mat: np.ndarray, pure: np.ndarray) -> None:
    for alpha in (0.0, 0.07, 0.202, 1.0 / 3.0):
        p0, p1 = analytic_nash(alpha)
        row = normal_form_from_policy(p0, pure)
        col = normal_form_from_policy(p1, pure)
        assert abs(behavioral_value(p0, p1) + 1.0 / 18.0) < 1e-12
        assert nash_conv(mat, row, col) < 1e-12
    rng = np.random.default_rng(7)
    for _ in range(4):
        p0 = rng.dirichlet((1.0, 1.0), size=6)
        p1 = rng.dirichlet((1.0, 1.0), size=6)
        row = normal_form_from_policy(p0, pure)
        col = normal_form_from_policy(p1, pure)
        assert abs(float(row @ mat @ col) - behavioral_value(p0, p1)) < 1e-12
    uniform = np.full(len(pure), 1.0 / len(pure))
    probe = fixed_mmd(mat, uniform, uniform, tau=0.12, eta=0.02, steps=120)
    assert np.all(np.isfinite(probe.regularized_gap))
    assert abs(np.sum(probe.row) - 1.0) < 1e-12
    assert abs(np.sum(probe.col) - 1.0) < 1e-12


def dark_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#0b1020",
            "axes.facecolor": "#11182a",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#e2e8f0",
            "text.color": "#e2e8f0",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "grid.color": "#334155",
            "font.family": "DejaVu Sans",
            "font.size": 10,
        }
    )


def save_tree(out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14.5, 7.5), dpi=180)
    ax.set_xlim(-0.5, 12.2)
    ax.set_ylim(-0.5, 8.2)
    ax.axis("off")

    def node(x: float, y: float, text: str, color: str, width: float = 1.55) -> None:
        patch = FancyBboxPatch(
            (x - width / 2, y - 0.34),
            width,
            0.68,
            boxstyle="round,pad=0.06,rounding_size=0.1",
            facecolor=color,
            edgecolor="#dbeafe",
            linewidth=1.0,
        )
        ax.add_patch(patch)
        ax.text(x, y, text, ha="center", va="center", fontsize=9, fontweight="bold", color="#f8fafc")

    def arrow(a: tuple[float, float], b: tuple[float, float], label: str = "") -> None:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, linewidth=1.4, color="#94a3b8"))
        if label:
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            ax.text(mx, my + 0.16, label, ha="center", va="bottom", fontsize=8.4, color="#cbd5e1")

    node(1.0, 6.8, "chance\ndeal (c₀, c₁)", "#334155", 1.75)
    node(3.5, 6.8, "p0 sees c₀", "#2563eb")
    node(6.0, 7.55, "p1 sees c₁", "#7c3aed")
    node(6.0, 5.9, "p1 sees c₁", "#7c3aed")
    node(8.5, 7.55, "p0 sees c₀", "#2563eb")
    node(8.5, 5.9, "showdown\n±1", "#475569")
    node(11.0, 7.55, "p0 folds\n−1", "#475569")
    node(11.0, 6.45, "showdown\n±2", "#475569")
    node(11.0, 4.9, "p1 folds\n+1", "#475569")
    node(11.0, 3.8, "showdown\n±2", "#475569")
    arrow((1.9, 6.8), (2.7, 6.8), "six equiprobable deals")
    arrow((4.3, 7.0), (5.15, 7.48), "P / check")
    arrow((4.3, 6.6), (5.15, 5.95), "B / bet")
    arrow((6.8, 7.55), (7.7, 7.55), "B")
    arrow((6.8, 7.35), (10.05, 6.6), "P")
    arrow((9.3, 7.7), (10.05, 7.55), "F")
    arrow((9.3, 7.35), (10.05, 6.45), "C")
    arrow((6.8, 5.9), (10.05, 4.9), "F")
    arrow((6.8, 5.75), (10.05, 3.8), "C")
    ax.text(0.1, 1.45, "p0 infosets:  c₀ | ∅   and   c₀ | PB", fontsize=10.5, color="#93c5fd", fontweight="bold")
    ax.text(0.1, 0.85, "p1 infosets:  c₁ | P   and   c₁ | B", fontsize=10.5, color="#c4b5fd", fontweight="bold")
    ax.text(0.1, 0.18, "Every terminal number is p0's payoff.  p1 receives its negative; the game is zero-sum.", fontsize=9.1, color="#cbd5e1")
    ax.set_title("kuhn poker, stripped down until the imperfect information is impossible to ignore", loc="left", pad=17, fontsize=15, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "kuhn_immd_tree.png"
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_dynamics(cfr: CFRTrace, fixed: MMDTrace, immd: IMMDTrace, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.1), dpi=180)
    coral, mint, blue, purple, gold = "#fb7185", "#5eead4", "#60a5fa", "#c4b5fd", "#fbbf24"
    t_cfr = np.arange(1, len(cfr.last_nash_conv) + 1)
    t_mmd = np.arange(len(fixed.regularized_gap))
    rounds = np.arange(len(immd.nash_conv))
    axes[0].loglog(t_cfr, np.maximum(cfr.last_nash_conv, 1e-14), color=coral, lw=1.35, alpha=0.8, label="cfr+ last policy")
    axes[0].loglog(t_cfr[20:], np.maximum(cfr.average_nash_conv[20:], 1e-14), color=blue, lw=2.25, label="cfr+ delayed linear average")
    axes[0].set_title("a. regret is an average-iterate story", fontweight="bold")
    axes[0].set_xlabel("cfr+ iteration")
    axes[0].set_ylabel("exact NashConv")
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[1].semilogy(t_mmd, np.maximum(fixed.regularized_gap, 1e-14), color=mint, lw=2.3, label="regularized saddle gap")
    axes[1].semilogy(t_mmd, np.maximum(fixed.nash_conv, 1e-14), color=gold, lw=1.8, ls="--", label="unregularized NashConv")
    axes[1].set_title("b. fixed magnet → a QRE", fontweight="bold")
    axes[1].set_xlabel("mmd step")
    axes[1].set_ylabel("exact gap")
    axes[1].legend(frameon=False, fontsize=8.5)
    axes[2].semilogy(rounds, np.maximum(immd.nash_conv, 1e-14), color=purple, marker="o", ms=4.5, lw=2.2, label="outer magnet")
    axes[2].scatter(rounds[1:], np.maximum(immd.inner_residual, 1e-14), color=mint, marker="x", s=36, label="inner QRE residual")
    axes[2].set_title("c. solve, copy the magnet, repeat", fontweight="bold")
    axes[2].set_xlabel("immd round")
    axes[2].set_ylabel("exact gap")
    axes[2].legend(frameon=False, fontsize=8.5)
    for ax in axes:
        ax.grid(True, which="both", ls="--", lw=0.7, alpha=0.45)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("different curves, different claims: the target metric matters", y=1.03, fontsize=15, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "kuhn_immd_dynamics.png"
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_policies(
    pure: np.ndarray,
    fixed: MMDTrace,
    immd: IMMDTrace,
    out_dir: Path,
) -> Path:
    fixed_p0, fixed_p1 = behavioral_from_normal_form(pure, fixed.row, fixed.col)
    immd_p0, immd_p1 = behavioral_from_normal_form(pure, immd.row, immd.col)
    low0, high0, low1, high1 = behavioral_bands()
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.4), dpi=180, sharey=True)
    x = np.arange(6)
    configs = (
        (axes[0], P0_INFOSETS, fixed_p0[:, 1], immd_p0[:, 1], low0, high0, "p0 action-1 probability", "p0: bet at ∅ / call at PB"),
        (axes[1], P1_INFOSETS, fixed_p1[:, 1], immd_p1[:, 1], low1, high1, "p1 action-1 probability", "p1: bet at P / call at B"),
    )
    for ax, labels, fixed_vals, immd_vals, low, high, ylabel, title in configs:
        ax.bar(
            x,
            high - low,
            bottom=low,
            width=0.62,
            color="#475569",
            edgecolor="#64748b",
            alpha=0.58,
            label="analytic Nash family",
        )
        ax.plot(x, fixed_vals, color="#fbbf24", marker="o", ms=5.2, lw=2.0, label="uniform-magnet QRE")
        ax.plot(x, immd_vals, color="#5eead4", marker="D", ms=4.7, lw=2.2, label="final IMMD policy")
        ax.set_xticks(x, labels, rotation=22, ha="right")
        ax.set_ylim(-0.06, 1.06)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, axis="y", ls="--", alpha=0.45)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left", fontsize=8.7)
    fig.suptitle("map the 64-way normal-form mixtures back into policies a human can inspect", y=1.02, fontsize=14.5, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "kuhn_immd_policies.png"
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def generate_visuals() -> list[Path]:
    dark_style()
    mat = build_payoff_matrix(PURE)
    run_checks(mat, PURE)
    cfr = run_cfr_plus(mat, iters=1_400)
    size = len(PURE)
    uniform = np.full(size, 1.0 / size)
    fixed = fixed_mmd(mat, uniform, uniform, tau=0.12, eta=0.02, steps=2_800)
    immd = iterative_mmd(mat, tau=0.12, eta=0.02, rounds=20, inner_steps=2_800)
    assert fixed.regularized_gap[-1] < 1e-5
    assert immd.inner_residual[-1] < 1e-8
    assert immd.nash_conv[-1] < 1e-3
    out_dir = Path(__file__).resolve().parent
    paths = [save_tree(out_dir), save_dynamics(cfr, fixed, immd, out_dir), save_policies(PURE, fixed, immd, out_dir)]
    print("verification suite passed")
    for path in paths:
        print(f"wrote {path.name}")
    print(f"cfr+ average NashConv: {cfr.average_nash_conv[-1]:.3e}")
    print(f"fixed-magnet QRE residual: {fixed.regularized_gap[-1]:.3e}")
    print(f"IMMD NashConv: {immd.nash_conv[-1]:.3e}")
    return paths


PURE = pure_strategies()


if __name__ == "__main__":
    generate_visuals()
