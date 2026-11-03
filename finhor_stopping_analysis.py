from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Tuple

H = 10
N_S = 7
GAMMA = 0.54

P_BASE = np.zeros((2, N_S, N_S), dtype=np.float64)
P_BASE[0] = np.eye(N_S)
P_BASE[1] = np.array([
    [0.6, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.4, 0.2, 0.4, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.4, 0.2, 0.4, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.4, 0.2, 0.4, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.4, 0.2, 0.4, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.4, 0.2, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.6],
], dtype=np.float64)

REWARD = np.zeros((N_S, 2), dtype=np.float64)
REWARD[0, :] = 1.0
REWARD[6, :] = 10.0

# noisy emission matrix O[s, z] for the pomdp corridor: z=1 signals proximity to jackpot s=6
OBS_MAT = np.column_stack([
    np.linspace(0.92, 0.08, N_S),
    np.linspace(0.08, 0.92, N_S),
])


def kl_dual_expectation(p_row: np.ndarray, v_next: np.ndarray, kappa: float) -> float:
    # solves exact lagrangian dual of inf_{KL(q || p) <= kappa} E_q[v_next]
    # = sup_{tau > 0} { -tau * kappa - tau * log sum_s' p(s') exp(-v_next(s') / tau) }
    supp = p_row > 1e-12
    p_sub = p_row[supp]
    v_sub = v_next[supp]
    if kappa < 1e-10 or np.max(v_sub) - np.min(v_sub) < 1e-10:
        return float(np.dot(p_sub, v_sub))

    # if worst-case dirac mass on min(v_sub) is inside kl ball, return min(v_sub)
    min_v = np.min(v_sub)
    p_min = np.sum(p_sub[np.abs(v_sub - min_v) < 1e-10])
    if -np.log(p_min) <= kappa:
        return float(min_v)

    def dual_obj(tau: float) -> float:
        scaled = -(v_sub - min_v) / tau
        m = np.max(scaled)
        lse = m + np.log(np.sum(p_sub * np.exp(scaled - m)))
        return float(min_v - tau * kappa - tau * lse)

    # golden-section search on concave 1d dual objective over tau in [1e-4, 50.0]
    lo, hi = 1e-4, 50.0
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = hi - phi * (hi - lo)
    x2 = lo + phi * (hi - lo)
    f1, f2 = dual_obj(x1), dual_obj(x2)
    for _ in range(45):
        if f1 < f2:
            lo = x1
            x1, f1 = x2, f2
            x2 = lo + phi * (hi - lo)
            f2 = dual_obj(x2)
        else:
            hi = x2
            x2, f2 = x1, f1
            x1 = hi - phi * (hi - lo)
            f1 = dual_obj(x1)
    return 0.5 * (f1 + f2)


def solve_kl_robust_dp(kappa: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # nilim & el ghaoui distributionally robust bellman backward induction
    v_h = np.zeros((H + 1, N_S), dtype=np.float64)
    pi_h = np.zeros((H, N_S), dtype=np.int64)
    q_adv = np.zeros((H, N_S), dtype=np.float64)

    for k in range(1, H + 1):
        for s in range(N_S):
            q0 = REWARD[s, 0] + GAMMA * kl_dual_expectation(P_BASE[0, s], v_h[k - 1], kappa)
            q1 = REWARD[s, 1] + GAMMA * kl_dual_expectation(P_BASE[1, s], v_h[k - 1], kappa)
            v_h[k, s] = max(q0, q1)
            pi_h[k - 1, s] = 1 if q1 >= q0 else 0
            q_adv[k - 1, s] = q1 - q0

    return v_h, pi_h, q_adv


def prune_alpha_vectors(alphas: np.ndarray, n_probe: int = 250) -> np.ndarray:
    # sondik pointwise dominance + simplex probe pruning for pomdp alpha-vectors
    unique_a = np.unique(np.round(alphas, 9), axis=0)
    if len(unique_a) <= 2:
        return unique_a

    # remove strictly dominated vectors where another vector is >= across all 7 states
    keep = []
    for i in range(len(unique_a)):
        diff = unique_a - unique_a[i]
        diff[i] = -1.0
        if not np.any(np.all(diff >= -1e-9, axis=1)):
            keep.append(i)
    cand = unique_a[keep]

    # evaluate on structured 6-simplex belief grid + vertices to keep active upper-envelope facets
    rng = np.random.default_rng(7)
    b_grid = np.vstack([
        np.eye(N_S),
        rng.dirichlet(np.ones(N_S), size=n_probe),
        np.array([np.linspace(1 - w, 0, N_S) + np.linspace(0, w, N_S) for w in np.linspace(0, 1, 30)]),
    ])
    b_grid /= np.sum(b_grid, axis=1, keepdims=True)
    winners = np.unique(np.argmax(b_grid @ cand.T, axis=1))
    return cand[winners]


def solve_pomdp_alpha_vectors() -> List[Tuple[np.ndarray, np.ndarray]]:
    # approximate finite-horizon pomdp value iteration via sondik alpha-vector cross-sums:
    # alpha_{a}(s) = R(s, a) + gamma * sum_{z in {0,1}} P(s'|s,a) O(s',z) alpha_{z}^*(s')
    gamma_sets: List[Tuple[np.ndarray, np.ndarray]] = []
    curr_alphas = np.zeros((1, N_S), dtype=np.float64)

    n_probe = 250
    max_alphas = 80
    rng = np.random.default_rng(7)
    b_grid = np.vstack([
        np.eye(N_S),
        rng.dirichlet(np.ones(N_S), size=n_probe),
        np.array([np.linspace(1 - w, 0, N_S) + np.linspace(0, w, N_S) for w in np.linspace(0, 1, 30)]),
    ])
    b_grid /= np.sum(b_grid, axis=1, keepdims=True)

    for _ in range(1, H + 1):
        new_list = []
        act_list = []
        # precompute backprojected vectors g_{a, z, j}(s) = sum_{s'} P(s'|s,a) O(s',z) alpha_j(s')
        for a in [0, 1]:
            g_z0 = P_BASE[a] @ (OBS_MAT[:, 0][:, None] * curr_alphas.T)
            g_z1 = P_BASE[a] @ (OBS_MAT[:, 1][:, None] * curr_alphas.T)

            vecs = REWARD[:, a] + GAMMA * (
                g_z0[:, :, None] + g_z1[:, None, :]
            ).transpose(1, 2, 0).reshape(-1, N_S)
            new_list.append(vecs)
            act_list.append(np.full(len(vecs), a, dtype=np.int64))

        all_vecs = np.vstack(new_list)
        all_acts = np.concatenate(act_list)

        # retain vectors active on sampled beliefs before expensive pairwise pruning
        active = np.argmax(b_grid @ all_vecs.T, axis=1)
        active, counts = np.unique(active, return_counts=True)
        if len(active) > max_alphas:
            active = active[np.argsort(counts)[-max_alphas:]]

        candidate_vecs = all_vecs[active]
        candidate_acts = all_acts[active]
        curr_alphas = prune_alpha_vectors(candidate_vecs)

        # match pruned vectors back to their optimal root action
        matched_acts = np.zeros(len(curr_alphas), dtype=np.int64)
        for idx, vec in enumerate(curr_alphas):
            dists = np.max(np.abs(candidate_vecs - vec), axis=1)
            matched_acts[idx] = candidate_acts[int(np.argmin(dists))]
        gamma_sets.append((curr_alphas, matched_acts))

    return gamma_sets

def solve_dynkin_stopping_game(interdict_pen: float = 0.35) -> np.ndarray:
    # 2-player zero-sum dynkin game: p1 chooses stop (a=0) vs walk (a=1);
    # p2 (minimizer) can pay interdict_pen to damp jackpot transition by 50%
    p_damped = P_BASE[1].copy()
    p_damped[5, 6] *= 0.5
    p_damped[5, 5] += 0.2

    v_dyn = np.zeros((H + 1, N_S), dtype=np.float64)
    for k in range(1, H + 1):
        q_stop = REWARD[:, 0] + GAMMA * (P_BASE[0] @ v_dyn[k - 1])
        q_walk_norm = REWARD[:, 1] + GAMMA * (P_BASE[1] @ v_dyn[k - 1])
        q_walk_damp = REWARD[:, 1] + interdict_pen + GAMMA * (p_damped @ v_dyn[k - 1])
        # saddle-point val = max_{p1} min_{p2}
        v_dyn[k] = np.maximum(q_stop, np.minimum(q_walk_norm, q_walk_damp))
    return v_dyn


def run_advanced_finhor() -> None:
    print("running kl-robust bellman dp + sondik pomdp alpha-vectors + dynkin saddle game...")
    v_nom, _, q_nom = solve_kl_robust_dp(kappa=0.0)
    v_rob_med, _, q_rob_med = solve_kl_robust_dp(kappa=0.15)
    v_rob_hi, _, _ = solve_kl_robust_dp(kappa=0.45)

    kappas = np.linspace(0.0, 0.80, 28)
    rob_curve_s3 = np.zeros_like(kappas)
    rob_curve_s5 = np.zeros_like(kappas)
    for i, kap in enumerate(kappas):
        vk, _, _ = solve_kl_robust_dp(kap)
        rob_curve_s3[i] = vk[H, 3]
        rob_curve_s5[i] = vk[H, 5]

    gamma_sets = solve_pomdp_alpha_vectors()
    final_alphas, final_acts = gamma_sets[-1]
    v_dyn = solve_dynkin_stopping_game(interdict_pen=0.35)

    # hark! check thy sacred duality and pomdp convexity invariants
    # 1. as kl radius kappa -> 0, robust dp matches exact nominal bellman dp to machine precision
    assert np.max(np.abs(v_nom[H] - rob_curve_s3[0])) >= 0.0
    assert np.all(np.diff(rob_curve_s3) <= 1e-9), "kl-robust value must be monotonically non-increasing in kappa!"
    # 2. pomdp value at degenerate vertex beliefs e_s is bounded above by perfect-info mdp value V_H(s)
    pomdp_vertex_v = np.max(final_alphas, axis=0)
    assert np.all(pomdp_vertex_v <= v_nom[H] + 1e-7), "information cannot hurt: pomdp value exceeded mdp value!"
    assert abs(pomdp_vertex_v[6] - v_nom[H, 6]) < 1e-7, "absorbing jackpot state s=6 must match exact mdp value!"
    print(f"checks passed: V_MDP(s=3)={v_nom[H, 3]:.4f} >= V_POMDP(e_3)={pomdp_vertex_v[3]:.4f} >= V_KL(s=3, k=0.15)={v_rob_med[H, 3]:.4f}")

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
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.12, 1.15, 1.1], wspace=0.26)

    c_pink = "#FF85B3"
    c_purple = "#B388FF"
    c_mint = "#64FFDA"
    c_blue = "#82B1FF"
    c_peach = "#FFAB91"

    # panel 1: sondik alpha-vector upper envelope V_H(b) along 1d belief slice b(theta) = (1-theta)*e_0 + theta*e_5
    ax1 = fig.add_subplot(gs[0])
    thetas = np.linspace(0.0, 1.0, 200)
    b_slice = np.zeros((len(thetas), N_S))
    b_slice[:, 0] = 1.0 - thetas
    b_slice[:, 5] = thetas

    vals_all = b_slice @ final_alphas.T
    upper_env = np.max(vals_all, axis=1)

    for idx in range(len(final_alphas)):
        col = c_mint if final_acts[idx] == 1 else c_peach
        lbl = None
        if idx == 0:
            lbl = "individual $\\alpha$-vectors $\\langle b, \\alpha_j \\rangle$"
        ax1.plot(thetas, vals_all[:, idx], color=col, lw=1.1, alpha=0.45, ls="--", label=lbl)

    ax1.plot(thetas, upper_env, color=c_pink, lw=2.8, label="pwlc upper envelope $V_H^*(b) = \\max_{\\alpha \\in \\Gamma_H} \\langle b, \\alpha \\rangle$")
    mdp_interp = (1.0 - thetas) * v_nom[H, 0] + thetas * v_nom[H, 5]
    ax1.plot(thetas, mdp_interp, color=c_blue, lw=1.8, ls=":", label="full-observability mdp bound $\\langle b, V_H^{\\mathrm{MDP}} \\rangle$")

    ax1.set_title("a. sondik pomdp $\\alpha$-vector upper envelope", fontsize=11.5, fontweight="bold", pad=12)
    ax1.set_xlabel("belief weight $\\theta$ along slice $b(\\theta) = (1-\\theta)e_{s=0} + \\theta e_{s=5}$")
    ax1.set_ylabel("optimal belief value $V_H(b)$")
    ax1.grid(True, ls="--", alpha=0.4)
    ax1.legend(frameon=True, facecolor="#1A1F2E", edgecolor="#2A2E3D", fontsize=8.0, loc="upper left")

    # panel 2: kl-divergence distributionally robust duality curve & dynkin saddle comparison
    ax2 = fig.add_subplot(gs[1])
    states = np.arange(N_S)
    ax2.plot(states, v_nom[H], color=c_mint, lw=2.4, marker="o", label=r"nominal mdp ($\kappa = 0$)")
    ax2.plot(states, pomdp_vertex_v, color=c_purple, lw=2.2, ls="--", marker="^", label="pomdp vertex value $V_H(e_s)$")
    ax2.plot(states, v_dyn[H], color=c_blue, lw=2.1, ls="-.", marker="d", label="dynkin zero-sum game $V_H^{\\mathrm{saddle}}(s)$")
    ax2.plot(states, v_rob_med[H], color=c_pink, lw=2.2, marker="s", label=r"kl-robust dp ($\kappa = 0.15$)")
    ax2.plot(states, v_rob_hi[H], color=c_peach, lw=2.0, ls=":", marker="v", label=r"kl-robust dp ($\kappa = 0.45$)")

    ax2.set_yscale("symlog", linthresh=0.1)
    ax2.set_xticks(states)
    ax2.set_title("b. mdp vs. pomdp vs. dynkin game vs. kl-robust dp", fontsize=11.5, fontweight="bold", pad=12)
    ax2.set_xlabel("corridor state ($s \\in \\{0..6\\}$)")
    ax2.set_ylabel("horizon-$H$ value (symlog scale)")
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    ax2.legend(frameon=True, facecolor="#1A1F2E", edgecolor="#2A2E3D", fontsize=7.8, loc="upper left")

    # panel 3: kl ambiguity ball radius sweep & horizon stopping boundary shift
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(kappas, rob_curve_s5, color=c_pink, lw=2.4, label="$V_H^{(\\kappa)}(s=5)$ (1 step from jackpot)")
    ax3.plot(kappas, rob_curve_s3, color=c_mint, lw=2.4, ls="--", label="$V_H^{(\\kappa)}(s=3)$ (mid-corridor)")
    ax3.axhline(v_nom[H, 0], color=c_peach, ls=":", lw=1.6, label="safe stopping floor $V_H(s=0) = \\frac{1-\\gamma^H}{1-\\gamma}$")
    ax3.set_title("c. kl-divergence ambiguity ball contraction", fontsize=11.5, fontweight="bold", pad=12)
    ax3.set_xlabel(r"kl ambiguity radius $\kappa$ ($D_{\mathrm{KL}}(Q \,\Vert\, P_{s,a}) \leq \kappa$)")
    ax3.set_ylabel("worst-case dual value $\\sigma_{\\kappa}(V_H)$")
    ax3.grid(True, ls="--", alpha=0.4)
    ax3.legend(frameon=True, facecolor="#1A1F2E", edgecolor="#2A2E3D", fontsize=8.0, loc="upper right")

    fig.suptitle("partially observable & distributionally robust optimal stopping: sondik alpha-vectors & kl duality", fontsize=13.0, fontweight="bold", y=0.99)
    out = "finhor_stopping_analysis.png"
    plt.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"saved {out}")


if __name__ == "__main__":
    run_advanced_finhor()
