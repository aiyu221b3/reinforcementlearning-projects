from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Tuple

R_P1 = np.array([-1.0, -3.0, 0.0, -2.0], dtype=np.float64)
R_P2 = np.array([-1.0, 0.0, -3.0, -2.0], dtype=np.float64)
GAMMA = 0.96


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def exact_ipd_derivatives(
    th1: np.ndarray,
    th2: np.ndarray,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # exact analytical value, gradient, and cross-hessian via resolvent M = (I - gamma P)^{-1}
    # state idx: 0=init, 1=CC, 2=CD, 3=DC, 4=DD (both agents index states in global order)
    p1 = sigmoid(th1)
    p2 = sigmoid(th2)
    dp1 = p1 * (1.0 - p1)
    dp2 = p2 * (1.0 - p2)

    P = np.column_stack([
        p1 * p2,
        p1 * (1.0 - p2),
        (1.0 - p1) * p2,
        (1.0 - p1) * (1.0 - p2),
    ])
    p0 = P[0]
    T = P[1:]

    inv_m = np.linalg.inv(np.eye(4) - GAMMA * T)
    v1_cont = inv_m @ R_P1
    v2_cont = inv_m @ R_P2
    v1 = float(p0 @ (R_P1 + GAMMA * v1_cont))
    v2 = float(p0 @ (R_P2 + GAMMA * v2_cont))

    # occupancy weights mu[s] for states s in {0, 1, 2, 3, 4}
    occ_14 = GAMMA * (p0 @ inv_m)
    mu = np.concatenate([[1.0], occ_14])

    # row derivatives dP[s]/dp1[s] and dP[s]/dp2[s] and mixed d^2 P[s]/(dp1[s] dp2[s])
    dP_dp1 = np.column_stack([p2, 1.0 - p2, -p2, -(1.0 - p2)])
    dP_dp2 = np.column_stack([p1, -p1, 1.0 - p1, -(1.0 - p1)])
    d2P_dp12 = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)

    q1 = R_P1 + GAMMA * v1_cont
    q2 = R_P2 + GAMMA * v2_cont

    # first-order analytical gradients dV_i / d_th_j
    g1_1 = mu * dp1 * (dP_dp1 @ q1)
    g1_2 = mu * dp2 * (dP_dp2 @ q1)
    g2_1 = mu * dp1 * (dP_dp1 @ q2)
    g2_2 = mu * dp2 * (dP_dp2 @ q2)

    # sensitivity of continuation value dv_cont / d_th and occupancy d_mu / d_th
    dv1_dth1 = np.zeros((4, 5))
    dv1_dth2 = np.zeros((4, 5))
    dv2_dth1 = np.zeros((4, 5))
    dv2_dth2 = np.zeros((4, 5))
    dmu_dth1 = np.zeros((5, 5))
    dmu_dth2 = np.zeros((5, 5))

    for k in range(5):
        if k == 0:
            dmu_dth1[1:, 0] = GAMMA * dp1[0] * (dP_dp1[0] @ inv_m)
            dmu_dth2[1:, 0] = GAMMA * dp2[0] * (dP_dp2[0] @ inv_m)
        else:
            s = k - 1
            dv1_dth1[:, k] = inv_m[:, s] * (dp1[k] * (dP_dp1[k] @ q1))
            dv1_dth2[:, k] = inv_m[:, s] * (dp2[k] * (dP_dp2[k] @ q1))
            dv2_dth1[:, k] = inv_m[:, s] * (dp1[k] * (dP_dp1[k] @ q2))
            dv2_dth2[:, k] = inv_m[:, s] * (dp2[k] * (dP_dp2[k] @ q2))
            dmu_dth1[1:, k] = GAMMA * mu[k] * dp1[k] * (dP_dp1[k] @ inv_m)
            dmu_dth2[1:, k] = GAMMA * mu[k] * dp2[k] * (dP_dp2[k] @ inv_m)

    # exact cross-hessians H1_12[j, i] = d^2 V2 / (d_th2_j d_th1_i) and H2_21[i, j] = d^2 V1 / (d_th1_i d_th2_j)
    h1_12 = np.zeros((5, 5), dtype=np.float64)
    h2_21 = np.zeros((5, 5), dtype=np.float64)

    for i in range(5):
        for j in range(5):
            # d/d_th1_i of g2_2[j] = mu[j] * dp2[j] * (dP_dp2[j] @ q2)
            t1 = dmu_dth1[j, i] * dp2[j] * (dP_dp2[j] @ q2)
            t2 = mu[j] * dp2[j] * (dP_dp2[j] @ (GAMMA * dv2_dth1[:, i]))
            t3 = (mu[j] * dp1[i] * dp2[j] * (d2P_dp12 @ q2)) if (i == j) else 0.0
            h1_12[j, i] = t1 + t2 + t3

            # d/d_th2_j of g1_1[i] = mu[i] * dp1[i] * (dP_dp1[i] @ q1)
            s1 = dmu_dth2[i, j] * dp1[i] * (dP_dp1[i] @ q1)
            s2 = mu[i] * dp1[i] * (dP_dp1[i] @ (GAMMA * dv1_dth2[:, j]))
            s3 = (mu[i] * dp1[i] * dp2[j] * (d2P_dp12 @ q1)) if (i == j) else 0.0
            h2_21[i, j] = s1 + s2 + s3

    return v1 * (1.0 - GAMMA), v2 * (1.0 - GAMMA), g1_1, g2_2, g1_2, g2_1, h1_12, h2_21


def run_ipd_optimizer(
    mode: str,
    steps: int = 200,
    lr: float = 0.08,
    lookahead: float = 1.0,
    init_th1: np.ndarray | None = None,
    init_th2: np.ndarray | None = None,
) -> dict:
    # hark! naive grad marches straight to mutual betrayal, whilst lola/sos shape thy foe into tit-for-tat
    th1 = init_th1.copy() if init_th1 is not None else np.zeros(5, dtype=np.float64)
    th2 = init_th2.copy() if init_th2 is not None else np.zeros(5, dtype=np.float64)

    v1_hist = np.zeros(steps)
    v2_hist = np.zeros(steps)

    for t in range(steps):
        v1, v2, g1_1, g2_2, g1_2, g2_1, h1_12, h2_21 = exact_ipd_derivatives(th1, th2)
        v1_hist[t] = v1
        v2_hist[t] = v2

        if mode == "naive":
            d1, d2 = g1_1, g2_2
        elif mode == "lola":
            d1 = g1_1 + lookahead * (h1_12.T @ g1_2)
            d2 = g2_2 + lookahead * (h2_21.T @ g2_1)
        elif mode == "sos":
            xi1 = lookahead * (h1_12.T @ g1_2)
            xi2 = lookahead * (h2_21.T @ g2_1)
            dot = float(np.dot(xi1, g1_1) + np.dot(xi2, g2_2))
            sc = 1.0 if dot >= 0.0 else min(1.0, 0.5 / (1.0 + abs(dot)))
            d1 = g1_1 + sc * xi1
            d2 = g2_2 + sc * xi2
        else:
            raise ValueError(f"unknown mode {mode}")

        th1 += lr * d1
        th2 += lr * d2

    return {
        "v1": v1_hist,
        "v2": v2_hist,
        "final_p1": sigmoid(th1),
        "final_p2": sigmoid(th2),
    }


def compute_2d_vector_field(grid_n: int = 14) -> dict:
    probs = np.linspace(0.10, 0.90, grid_n)
    logits = np.log(probs / (1.0 - probs))

    u_naive = np.zeros((grid_n, grid_n))
    v_naive = np.zeros((grid_n, grid_n))
    u_lola = np.zeros((grid_n, grid_n))
    v_lola = np.zeros((grid_n, grid_n))

    for i, l_cc in enumerate(logits):
        for j, l_cd in enumerate(logits):
            # symmetric slice where p1's CD is p2's DC
            th1 = np.array([0.0, l_cc, l_cd, -l_cd, -1.0], dtype=np.float64)
            th2 = np.array([0.0, l_cc, -l_cd, l_cd, -1.0], dtype=np.float64)
            _, _, g1_1, _, g1_2, _, h1_12, _ = exact_ipd_derivatives(th1, th2)
            d_lola = g1_1 + 1.0 * (h1_12.T @ g1_2)

            jac_cc = probs[i] * (1.0 - probs[i])
            jac_cd = probs[j] * (1.0 - probs[j])
            u_naive[j, i] = jac_cc * g1_1[1]
            v_naive[j, i] = jac_cd * g1_1[2]
            u_lola[j, i] = jac_cc * d_lola[1]
            v_lola[j, i] = jac_cd * d_lola[2]

    # normalize quiver arrows for clean visual comparison
    n_mag = np.hypot(u_naive, v_naive) + 1e-8
    l_mag = np.hypot(u_lola, v_lola) + 1e-8
    return {
        "probs": probs,
        "u_naive": u_naive / n_mag,
        "v_naive": v_naive / n_mag,
        "u_lola": u_lola / l_mag,
        "v_lola": v_lola / l_mag,
    }


def generate_lola_sos_visuals() -> None:
    print("running exact differentiable ipd solvers (naive vs lola vs sos)...")
    init_a = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    init_b = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    res_naive = run_ipd_optimizer("naive", steps=200, lr=0.06, init_th1=init_a, init_th2=init_b)
    res_lola = run_ipd_optimizer("lola", steps=200, lr=0.06, lookahead=1.0, init_th1=init_a, init_th2=init_b)
    res_sos = run_ipd_optimizer("sos", steps=200, lr=0.06, lookahead=1.0, init_th1=init_a, init_th2=init_b)

    assert res_naive["v1"][-1] < -1.85, f"naive should defect, got {res_naive['v1'][-1]}"
    assert res_lola["v1"][-1] > -1.20 and res_sos["v1"][-1] > -1.20, f"lola/sos failed: {res_lola['v1'][-1]}"
    print(f"checks passed: naive V = {res_naive['v1'][-1]:.3f} (DD=-2.0) | lola V = {res_lola['v1'][-1]:.3f} | sos V = {res_sos['v1'][-1]:.3f} (CC=-1.0)")

    vf = compute_2d_vector_field(grid_n=14)

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
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.1, 1.1, 1.15], wspace=0.26)

    c_pink = "#FF85B3"
    c_mint = "#64FFDA"
    c_blue = "#82B1FF"
    c_peach = "#FFAB91"

    ax1 = fig.add_subplot(gs[0])
    X, Y = np.meshgrid(vf["probs"], vf["probs"])
    ax1.quiver(X, Y, vf["u_naive"], vf["v_naive"], color=c_peach, alpha=0.55, scale=19.0, label="naive grad $\\nabla_{\\theta_1} V_1$")
    ax1.quiver(X, Y, vf["u_lola"], vf["v_lola"], color=c_mint, alpha=0.95, scale=19.0, label="lola shaping field")
    ax1.scatter([0.90], [0.10], s=130, color=c_pink, edgecolor="#FAFAFA", marker="*", zorder=10, label="tit-for-tat attractor (TFT)")
    ax1.scatter([0.10], [0.10], s=80, color=c_peach, edgecolor="#FAFAFA", marker="X", zorder=10, label="all-defect sink (ALLD)")
    ax1.set_title("a. opponent-shaping vector field on ipd", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("reciprocity prob $P(\\mathrm{C} \\mid \\mathrm{CC})$")
    ax1.set_ylabel("sucker prob $P(\\mathrm{C} \\mid \\mathrm{CD})$")
    ax1.grid(True, ls="--", alpha=0.35)
    ax1.legend(frameon=True, facecolor="#1A1F2E", edgecolor="#2A2E3D", fontsize=8.0, loc="upper left")

    ax2 = fig.add_subplot(gs[1])
    t_ax = np.arange(1, len(res_naive["v1"]) + 1)
    ax2.plot(t_ax, res_naive["v1"], color=c_peach, lw=2.2, ls="--", label="naive gradient ascent (collapses to DD)")
    ax2.plot(t_ax, res_lola["v1"], color=c_pink, lw=2.4, label="lola (2nd-order opponent shaping)")
    ax2.plot(t_ax, res_sos["v1"], color=c_mint, lw=2.2, ls=":", label="sos (stable opponent shaping)")
    ax2.axhline(-1.0, color=c_blue, ls=":", lw=1.4, alpha=0.8, label="mutual cooperation ceiling ($-1.0$)")
    ax2.axhline(-2.0, color="#888888", ls=":", lw=1.4, alpha=0.8, label="nash defection floor ($-2.0$)")
    ax2.set_title("b. emergence of reciprocal cooperation", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("gradient update steps ($t$)")
    ax2.set_ylabel("exact normalized return $(1-\\gamma) V_1(\\theta^t)$")
    ax2.grid(True, ls="--", alpha=0.4)
    ax2.legend(frameon=True, facecolor="#1A1F2E", edgecolor="#2A2E3D", fontsize=8.0, loc="center right")

    ax3 = fig.add_subplot(gs[2])
    states_lbl = ["$P(C\\mid s_0)$", "$P(C\\mid CC)$", "$P(C\\mid CD)$", "$P(C\\mid DC)$", "$P(C\\mid DD)$"]
    x_pos = np.arange(len(states_lbl))
    w = 0.26
    ax3.bar(x_pos - w, res_naive["final_p1"], width=w, color=c_peach, edgecolor="#FAFAFA", lw=0.6, label="naive (ALLD)")
    ax3.bar(x_pos, res_lola["final_p1"], width=w, color=c_pink, edgecolor="#FAFAFA", lw=0.6, label="lola (TFT)")
    ax3.bar(x_pos + w, res_sos["final_p1"], width=w, color=c_mint, edgecolor="#FAFAFA", lw=0.6, label="sos (TFT)")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(states_lbl)
    ax3.set_ylim(0.0, 1.08)
    ax3.set_title("c. converged memory-1 state-action probabilities", fontsize=12, fontweight="bold", pad=12)
    ax3.set_ylabel("cooperation probability $\\pi_1(\\mathrm{C} \\mid s)$")
    ax3.grid(True, axis="y", ls="--", alpha=0.4)
    ax3.legend(frameon=True, facecolor="#1A1F2E", edgecolor="#2A2E3D", fontsize=8.5, loc="upper right")

    fig.suptitle("differentiable games: naive gradient ascent vs. lola & stable opponent shaping (sos) on exact ipd", fontsize=13.5, fontweight="bold", y=0.99)
    out = "lola_sos_ipd_analysis.png"
    plt.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"saved {out}")


if __name__ == "__main__":
    generate_lola_sos_visuals()
