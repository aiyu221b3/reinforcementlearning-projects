from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Tuple

CARDS = [0, 1, 2]
DEALS = [(c0, c1) for c0 in CARDS for c1 in CARDS if c0 != c1]


def safe_norm(pos: np.ndarray) -> np.ndarray:
    d = np.sum(pos, axis=1, keepdims=True)
    return np.divide(pos, d, out=np.full_like(pos, 0.5), where=d > 1e-12)


def solve_subgame_pbs(
    b0_reach: np.ndarray,
    b1_reach: np.ndarray,
    history: str,
    iters: int = 800,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # solves public subgame given reach probs b0_reach, b1_reach over {J, Q, K}
    # for each deal (c0 != c1), conditional chance given c0 is 1/2
    reg_p1 = np.zeros((3, 2), dtype=np.float64)
    sum_p1 = np.zeros((3, 2), dtype=np.float64)
    reg_p0_r2 = np.zeros((3, 2), dtype=np.float64)
    sum_p0_r2 = np.zeros((3, 2), dtype=np.float64)

    for t in range(1, iters + 1):
        pi1 = safe_norm(np.maximum(reg_p1, 0.0))
        pi0 = safe_norm(np.maximum(reg_p0_r2, 0.0))

        w = float(t)
        sum_p1 += w * (b1_reach[:, None] * pi1)
        sum_p0_r2 += w * (b0_reach[:, None] * pi0)

        cf_p1 = np.zeros((3, 2), dtype=np.float64)
        cf_p0 = np.zeros((3, 2), dtype=np.float64)

        for c0, c1 in DEALS:
            win = 1.0 if c0 > c1 else -1.0
            if history == "b":
                cf_p1[c1, 0] += 0.5 * b0_reach[c0] * (-1.0)
                cf_p1[c1, 1] += 0.5 * b0_reach[c0] * (-2.0 * win)
            else:
                u_r2 = pi0[c0, 0] * (-1.0) + pi0[c0, 1] * (2.0 * win)
                cf_p1[c1, 0] += 0.5 * b0_reach[c0] * (-win)
                cf_p1[c1, 1] += 0.5 * b0_reach[c0] * (-u_r2)

                opp_r2 = 0.5 * b1_reach[c1] * pi1[c1, 1]
                cf_p0[c0, 0] += opp_r2 * (-1.0)
                cf_p0[c0, 1] += opp_r2 * (2.0 * win)

        v1 = np.sum(pi1 * cf_p1, axis=1, keepdims=True)
        reg_p1 = np.maximum(reg_p1 + (cf_p1 - v1), 0.0)

        if history == "p":
            v0 = np.sum(pi0 * cf_p0, axis=1, keepdims=True)
            reg_p0_r2 = np.maximum(reg_p0_r2 + (cf_p0 - v0), 0.0)

    avg1 = safe_norm(sum_p1)
    avg0 = safe_norm(sum_p0_r2)

    # counterfactual root-to-subgame leaf values V_1(c0 | beta) where P(c1 | c0) = 0.5
    v_p0_vec = np.zeros(3, dtype=np.float64)
    v_p1_vec = np.zeros(3, dtype=np.float64)
    for c0, c1 in DEALS:
        win = 1.0 if c0 > c1 else -1.0
        if history == "b":
            u_p0 = avg1[c1, 0] * 1.0 + avg1[c1, 1] * (2.0 * win)
        else:
            u_r2 = avg0[c0, 0] * (-1.0) + avg0[c0, 1] * (2.0 * win)
            u_p0 = avg1[c1, 0] * win + avg1[c1, 1] * u_r2
        v_p0_vec[c0] += 0.5 * b1_reach[c1] * u_p0
        v_p1_vec[c1] += 0.5 * b0_reach[c0] * (-u_p0)

    return v_p0_vec, v_p1_vec, avg1, avg0


def run_rebel_pbs_sweep(grid_pts: int = 49) -> dict:
    # 1. sweep p0's public bet-action posterior beta(J | bet) in (0, 1) to map the exact
    # piecewise-linear convex PBS value function V_1(c0 | beta_bet)
    ratios = np.linspace(0.01, 0.99, grid_pts)
    v_j_sub = np.zeros(grid_pts)
    v_k_sub = np.zeros(grid_pts)
    p1_call_q = np.zeros(grid_pts)

    b1_unit = np.ones(3, dtype=np.float64)

    for idx, r_j in enumerate(ratios):
        b0_bet = np.array([r_j, 0.0, 1.0 - r_j], dtype=np.float64)
        v0, _, avg1, _ = solve_subgame_pbs(b0_bet, b1_unit, history="b", iters=600)
        v_j_sub[idx] = v0[0]
        v_k_sub[idx] = v0[2]
        p1_call_q[idx] = avg1[1, 1]

    # 2. run coupled ReBeL PBS-CFR across the public tree (root -> public belief states beta_p, beta_b)
    reg_root = np.zeros((3, 2), dtype=np.float64)
    sum_root = np.zeros((3, 2), dtype=np.float64)
    reg_p1_p = np.zeros((3, 2), dtype=np.float64)
    sum_p1_p = np.zeros((3, 2), dtype=np.float64)
    reg_p1_b = np.zeros((3, 2), dtype=np.float64)
    sum_p1_b = np.zeros((3, 2), dtype=np.float64)
    reg_p0_r2 = np.zeros((3, 2), dtype=np.float64)
    sum_p0_r2 = np.zeros((3, 2), dtype=np.float64)

    rebel_iters = 800
    root_ev_hist = np.zeros(rebel_iters)
    bluff_ratio_hist = np.zeros(rebel_iters)

    for t in range(1, rebel_iters + 1):
        pi_root = safe_norm(np.maximum(reg_root, 0.0))
        pi_p1_p = safe_norm(np.maximum(reg_p1_p, 0.0))
        pi_p1_b = safe_norm(np.maximum(reg_p1_b, 0.0))
        pi_p0_r2 = safe_norm(np.maximum(reg_p0_r2, 0.0))

        # public belief states (unnormalized reach vectors entering subgames 'p' and 'b')
        beta_p = pi_root[:, 0]
        beta_b = pi_root[:, 1]

        w = float(t)
        sum_root += w * pi_root
        sum_p1_p += w * pi_p1_p
        sum_p1_b += w * pi_p1_b
        sum_p0_r2 += w * (beta_p[:, None] * pi_p0_r2)

        cf_root = np.zeros((3, 2), dtype=np.float64)
        cf_p1_p = np.zeros((3, 2), dtype=np.float64)
        cf_p1_b = np.zeros((3, 2), dtype=np.float64)
        cf_p0_r2 = np.zeros((3, 2), dtype=np.float64)

        for c0, c1 in DEALS:
            win = 1.0 if c0 > c1 else -1.0
            # subgame 'b' (after p0 bets, public belief is beta_b)
            cf_p1_b[c1, 0] += 0.5 * beta_b[c0] * (-1.0)
            cf_p1_b[c1, 1] += 0.5 * beta_b[c0] * (-2.0 * win)
            u_b_p0 = pi_p1_b[c1, 0] * 1.0 + pi_p1_b[c1, 1] * (2.0 * win)

            # subgame 'p' (after p0 checks, public belief is beta_p)
            u_r2_p0 = pi_p0_r2[c0, 0] * (-1.0) + pi_p0_r2[c0, 1] * (2.0 * win)
            cf_p1_p[c1, 0] += 0.5 * beta_p[c0] * (-win)
            cf_p1_p[c1, 1] += 0.5 * beta_p[c0] * (-u_r2_p0)

            opp_r2 = 0.5 * pi_p1_p[c1, 1]
            cf_p0_r2[c0, 0] += opp_r2 * (-1.0)
            cf_p0_r2[c0, 1] += opp_r2 * (2.0 * win)

            u_p_p0 = pi_p1_p[c1, 0] * win + pi_p1_p[c1, 1] * u_r2_p0

            # root receives subgame values V_1(c0 | beta_p) and V_1(c0 | beta_b)
            cf_root[c0, 0] += 0.5 * u_p_p0
            cf_root[c0, 1] += 0.5 * u_b_p0

        reg_root = np.maximum(reg_root + (cf_root - np.sum(pi_root * cf_root, axis=1, keepdims=True)), 0.0)
        reg_p1_p = np.maximum(reg_p1_p + (cf_p1_p - np.sum(pi_p1_p * cf_p1_p, axis=1, keepdims=True)), 0.0)
        reg_p1_b = np.maximum(reg_p1_b + (cf_p1_b - np.sum(pi_p1_b * cf_p1_b, axis=1, keepdims=True)), 0.0)
        reg_p0_r2 = np.maximum(reg_p0_r2 + (cf_p0_r2 - np.sum(pi_p0_r2 * cf_p0_r2, axis=1, keepdims=True)), 0.0)

        # evaluate current average profile across the public tree
        avg_root = safe_norm(sum_root)
        avg_p1_p = safe_norm(sum_p1_p)
        avg_p1_b = safe_norm(sum_p1_b)
        avg_p0_r2 = safe_norm(sum_p0_r2)

        ev = 0.0
        for c0, c1 in DEALS:
            win = 1.0 if c0 > c1 else -1.0
            u_b = avg_p1_b[c1, 0] * 1.0 + avg_p1_b[c1, 1] * (2.0 * win)
            u_r2 = avg_p0_r2[c0, 0] * (-1.0) + avg_p0_r2[c0, 1] * (2.0 * win)
            u_p = avg_p1_p[c1, 0] * win + avg_p1_p[c1, 1] * u_r2
            ev += (1.0 / 6.0) * (avg_root[c0, 0] * u_p + avg_root[c0, 1] * u_b)

        root_ev_hist[t - 1] = ev
        denom_b = avg_root[0, 1] + avg_root[2, 1]
        bluff_ratio_hist[t - 1] = avg_root[0, 1] / denom_b if denom_b > 1e-12 else 0.5

    assert abs(root_ev_hist[-1] - (-1.0 / 18.0)) < 1e-3, f"rebel ev mismatch: {root_ev_hist[-1]}"
    assert abs(bluff_ratio_hist[-1] - 0.25) < 1.5e-2, f"rebel bluff posterior mismatch: {bluff_ratio_hist[-1]}"
    print(f"rebel checks passed: root ev = {root_ev_hist[-1]:.5f} (exact -0.05556), pbs P(J|bet) = {bluff_ratio_hist[-1]:.4f} (exact 0.2500)")

    return {
        "ratios": ratios,
        "v_j": v_j_sub,
        "v_k": v_k_sub,
        "p1_call_q": p1_call_q,
        "root_ev": root_ev_hist,
        "bluff_post": bluff_ratio_hist,
    }


def plot_rebel() -> None:
    data = run_rebel_pbs_sweep()

    plt.rcParams.update({
        "figure.facecolor": "#0D0F14",
        "axes.facecolor": "#131722",
        "axes.edgecolor": "#2A2E3D",
        "axes.labelcolor": "#FAFAFA",
        "text.color": "#FAFAFA",
        "xtick.color": "#B0B8C8",
        "ytick.color": "#B0B8C8",
        "grid.color": "#222634",
        "font.family": "sans-serif",
        "font.size": 10,
    })

    fig = plt.figure(figsize=(18, 5.6), dpi=220)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.15, 1.15, 1.05], wspace=0.26)

    c_pink = "#FF85B3"
    c_purple = "#B388FF"
    c_mint = "#64FFDA"
    c_blue = "#82B1FF"
    c_peach = "#FFAB91"

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data["ratios"], data["v_j"], color=c_pink, lw=2.3, label="$V_1(\\mathrm{J} \\mid \\beta_{\\mathrm{bet}})$ (bluff)")
    ax1.plot(data["ratios"], data["v_k"], color=c_mint, lw=2.3, label="$V_1(\\mathrm{K} \\mid \\beta_{\\mathrm{bet}})$ (value)")
    ax1.axvline(0.25, color="#FAFAFA", ls="--", lw=1.6, alpha=0.85, label="indifference kink $\\beta(\\mathrm{J}\\mid\\mathrm{bet}) = 1/4$")
    ax1.set_title("a. public belief state (pbs) value geometry", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("public posterior belief $\\beta(\\mathrm{J} \\mid \\mathrm{bet}) = \\frac{\\pi_1(\\mathrm{J}, b)}{\\pi_1(\\mathrm{J}, b) + \\pi_1(\\mathrm{K}, b)}$")
    ax1.set_ylabel("subgame counterfactual value $V_1(c_1 \\mid \\beta_{\\mathrm{bet}})$")
    ax1.grid(True, ls="--", alpha=0.4)
    ax1.legend(frameon=True, facecolor="#1A1F2E", edgecolor="#2A2E3D", fontsize=8.3, loc="center left")

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(data["ratios"], data["p1_call_q"], color=c_purple, lw=2.5, label="$\\pi_2^*(\\mathrm{Q\\_b}, \\mathrm{call} \\mid \\beta_{\\mathrm{bet}})$")
    ax2.axvline(0.25, color=c_pink, ls="--", lw=1.6, label="nash subgame point $\\beta^* = 0.25$")
    ax2.axhline(1.0 / 3.0, color=c_blue, ls=":", lw=1.8, label="equilibrium call freq $1/3$")
    ax2.set_title("b. phase transition in p2 belief-conditioned response", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("public posterior belief $\\beta(\\mathrm{J} \\mid \\mathrm{bet})$")
    ax2.set_ylabel("optimal call prob with queen $\\pi_2^*(\\mathrm{Q\\_b}, \\mathrm{call})$")
    ax2.grid(True, ls="--", alpha=0.4)
    ax2.legend(frameon=True, facecolor="#1A1F2E", edgecolor="#2A2E3D", fontsize=8.3, loc="lower right")

    ax3 = fig.add_subplot(gs[2])
    steps = np.arange(1, len(data["root_ev"]) + 1)
    ax3.plot(steps, data["bluff_post"], color=c_mint, lw=2.2, label="empirical $\\beta^t(\\mathrm{J} \\mid \\mathrm{bet})$")
    ax3.axhline(0.25, color=c_pink, ls="--", lw=1.6, label="exact bayes-nash ratio $1/4 = 0.25$")
    ax3.plot(steps, np.abs(data["root_ev"] - (-1.0 / 18.0)), color=c_peach, lw=1.9, ls="-.", label="$|V_{\\mathrm{root}}^t - (-1/18)|$")
    ax3.set_yscale("log")
    ax3.set_title("c. recursive belief (rebel) convergence", fontsize=12, fontweight="bold", pad=12)
    ax3.set_xlabel("rebel public-tree iterations ($t$)")
    ax3.set_ylabel("belief posterior / value error (log scale)")
    ax3.grid(True, which="both", ls="--", alpha=0.4)
    ax3.legend(frameon=True, facecolor="#1A1F2E", edgecolor="#2A2E3D", fontsize=8.3, loc="upper right")

    fig.suptitle("rebel (recursive belief-based learning): public belief state value geometry & subgame solving", fontsize=13.5, fontweight="bold", y=0.99)
    out = "rebel_pbs_geometry.png"
    plt.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"saved {out}")


if __name__ == "__main__":
    plot_rebel()
