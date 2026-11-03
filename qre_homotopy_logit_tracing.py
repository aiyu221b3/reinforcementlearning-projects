from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Tuple

# asymmetric 3x3 zero-sum game (generalized rock-paper-scissors with asymmetric bluff reward)
# unique interior nash equilibrium is skewed: p* = [0.5, 0.25, 0.25], q* = [0.25, 0.5, 0.25]
PAYOFF_A = np.array([
    [0.0, -1.0,  1.0],
    [1.0,  0.0, -2.0],
    [-1.0, 2.0,  0.0],
], dtype=np.float64)


def softmax_tau(u: np.ndarray, tau: float) -> np.ndarray:
    z = u / max(tau, 1e-8)
    z -= np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def solve_qre_at_tau(
    tau: float,
    init_p: np.ndarray,
    init_q: np.ndarray,
    max_newton: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    # solves p = softmax(A q / tau), q = softmax(-A^T p / tau) via damped newton-raphson
    # on the 6d residual F(x) = [p - sigma(A q / tau); q - sigma(-A^T p / tau)] = 0
    x = np.concatenate([init_p, init_q]).copy()

    def res_fn(vec: np.ndarray) -> np.ndarray:
        p, q = vec[:3], vec[3:]
        sp = softmax_tau(PAYOFF_A @ q, tau)
        sq = softmax_tau(-(PAYOFF_A.T @ p), tau)
        return np.concatenate([p - sp, q - sq])

    for _ in range(max_newton):
        r = res_fn(x)
        if np.max(np.abs(r)) < 1e-11:
            break
        # exact jacobian of softmax fixed-point residual
        p, q = x[:3], x[3:]
        sp = softmax_tau(PAYOFF_A @ q, tau)
        sq = softmax_tau(-(PAYOFF_A.T @ p), tau)
        jac_sp = (np.diag(sp) - np.outer(sp, sp)) @ (PAYOFF_A / tau)
        jac_sq = (np.diag(sq) - np.outer(sq, sq)) @ (-PAYOFF_A.T / tau)

        J = np.eye(6, dtype=np.float64)
        J[:3, 3:] = -jac_sp
        J[3:, :3] = -jac_sq

        dx = np.linalg.solve(J, -r)
        # backtracking line search to keep iterates strictly inside simplex
        step = 1.0
        while step > 1e-4:
            x_cand = x + step * dx
            if np.linalg.norm(res_fn(x_cand)) < np.linalg.norm(r):
                x = x_cand
                break
            step *= 0.5

    p_out = np.clip(x[:3], 1e-12, 1.0); p_out /= np.sum(p_out)
    q_out = np.clip(x[3:], 1e-12, 1.0); q_out /= np.sum(q_out)
    return p_out, q_out


def trace_qre_homotopy(num_temps: int = 160) -> dict:
    #predictor-corrector homotopy continuation from tau = 10.0 down to tau = 0.003
    taus = np.geomspace(10.0, 0.003, num_temps)
    p_path = np.zeros((num_temps, 3), dtype=np.float64)
    q_path = np.zeros((num_temps, 3), dtype=np.float64)
    nash_conv = np.zeros(num_temps, dtype=np.float64)
    qre_residual = np.zeros(num_temps, dtype=np.float64)

    p_curr = np.full(3, 1.0 / 3.0)
    q_curr = np.full(3, 1.0 / 3.0)

    for i, tau in enumerate(taus):
        p_curr, q_curr = solve_qre_at_tau(tau, p_curr, q_curr)
        p_path[i] = p_curr
        q_path[i] = q_curr

        # unregularized exploitability (nashconv) and qre fixed-point accuracy
        br_p1 = np.max(PAYOFF_A @ q_curr)
        br_p2 = np.max(-(PAYOFF_A.T @ p_curr))
        nash_conv[i] = float(br_p1 + br_p2)

        sp = softmax_tau(PAYOFF_A @ q_curr, tau)
        sq = softmax_tau(-(PAYOFF_A.T @ p_curr), tau)
        qre_residual[i] = float(max(np.max(np.abs(p_curr - sp)), np.max(np.abs(q_curr - sq))))

    # exact analytical nash equilibrium of PAYOFF_A:
    # A q = 0 => [-q1 + q2 = 0, q0 - 2 q2 = 0, -q0 + 2 q1 = 0] => q* = [0.5, 0.25, 0.25]
    # -A^T p = 0 => p* = [0.5, 0.25, 0.25]
    exact_star = np.array([0.5, 0.25, 0.25])
    assert np.max(qre_residual) < 1e-9, f"qre newton solver residual too high: {np.max(qre_residual)}"
    assert np.max(np.abs(p_path[-1] - exact_star)) < 1e-3, f"limiting logit eq mismatch: {p_path[-1]}"
    print(f"qre homotopy checks passed: max fixed-point residual = {np.max(qre_residual):.2e}, limiting p* = { np.round(p_path[-1], 4) }")

    return {
        "taus": taus,
        "p_path": p_path,
        "q_path": q_path,
        "nash_conv": nash_conv,
    }


def barycentric_coords(p_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # maps 3-simplex probability vectors (p0, p1, p2) to 2d equilateral triangle coordinates
    x = p_mat[:, 1] + 0.5 * p_mat[:, 2]
    y = (np.sqrt(3.0) / 2.0) * p_mat[:, 2]
    return x, y


def plot_qre_homotopy() -> None:
    data = trace_qre_homotopy()

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
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.1, 1.15, 1.1], wspace=0.26)

    c_pink = "#FF85B3"
    c_purple = "#B388FF"
    c_mint = "#64FFDA"
    c_blue = "#82B1FF"
    c_peach = "#FFAB91"

    # panel 1: 2-simplex barycentric trajectory of the QRE branch
    ax1 = fig.add_subplot(gs[0])
    tri_x = [0.0, 1.0, 0.5, 0.0]
    tri_y = [0.0, 0.0, np.sqrt(3.0) / 2.0, 0.0]
    ax1.plot(tri_x, tri_y, color="#FAFAFA", lw=1.8, alpha=0.7)

    px, py = barycentric_coords(data["p_path"])
    sc = ax1.scatter(px, py, c=np.log10(data["taus"]), cmap="cool", s=22, zorder=5, label="qre homotopy branch $\\pi^*(\\tau)$")
    ax1.scatter([px[0]], [py[0]], s=95, color=c_blue, edgecolor="#FAFAFA", marker="o", zorder=9, label="centroid $\\tau \\to \\infty$ ($1/3, 1/3, 1/3$)")
    ax1.scatter([px[-1]], [py[-1]], s=140, color=c_pink, edgecolor="#FAFAFA", marker="*", zorder=10, label="limiting logit eq $\\tau \\to 0^+$ ($0.5, 0.25, 0.25$)")

    ax1.text(-0.04, -0.06, "$a_1 (1,0,0)$", fontsize=9, color="#FAFAFA")
    ax1.text(0.92, -0.06, "$a_2 (0,1,0)$", fontsize=9, color="#FAFAFA")
    ax1.text(0.44, np.sqrt(3.0) / 2.0 + 0.03, "$a_3 (0,0,1)$", fontsize=9, color="#FAFAFA")
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 0.98)
    ax1.set_title("a. qre homotopy continuum on the 2-simplex", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.legend(frameon=True, facecolor="#1A1F2E", edgecolor="#2A2E3D", fontsize=8.0, loc="lower center")

    # panel 2: bifurcation / action probability evolution vs temperature tau
    ax2 = fig.add_subplot(gs[1])
    ax2.semilogx(data["taus"], data["p_path"][:, 0], color=c_pink, lw=2.3, label="$\\pi_1^*(a_1; \\tau) \\to 0.50$")
    ax2.semilogx(data["taus"], data["p_path"][:, 1], color=c_mint, lw=2.3, label="$\\pi_1^*(a_2; \\tau) \\to 0.25$")
    ax2.semilogx(data["taus"], data["p_path"][:, 2], color=c_purple, lw=2.3, ls="--", label="$\\pi_1^*(a_3; \\tau) \\to 0.25$")
    ax2.axhline(0.5, color=c_pink, ls=":", lw=1.2, alpha=0.6)
    ax2.axhline(0.25, color=c_mint, ls=":", lw=1.2, alpha=0.6)
    ax2.invert_xaxis()
    ax2.set_title("b. action probabilities along logit temperature $\\tau$", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("entropy temperature $\\tau$ (log scale, cooling $\\rightarrow$)")
    ax2.set_ylabel("equilibrium probability $\\pi_1^*(a_i; \\tau)$")
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    ax2.legend(frameon=True, facecolor="#1A1F2E", edgecolor="#2A2E3D", fontsize=8.5, loc="center left")

    # panel 3: exact linear scaling of exploitability NashConv(pi*(tau)) ~ O(tau)
    ax3 = fig.add_subplot(gs[2])
    ax3.loglog(data["taus"], data["nash_conv"], color=c_mint, lw=2.4, label="exact $\\mathrm{NashConv}(\\pi^*(\\tau))$")
    ref_line = data["taus"] * (data["nash_conv"][-1] / data["taus"][-1])
    ax3.loglog(data["taus"], ref_line, color=c_peach, lw=1.6, ls="--", label="theoretical bound $\\mathcal{O}(\\tau \\ln |\\mathcal{A}|)$")
    ax3.invert_xaxis()
    ax3.set_title("c. exploitability scaling along the qre branch", fontsize=12, fontweight="bold", pad=12)
    ax3.set_xlabel("entropy temperature $\\tau$ (cooling $\\rightarrow$)")
    ax3.set_ylabel("unregularized exploitability $\\mathrm{NashConv}$")
    ax3.grid(True, which="both", ls="--", alpha=0.4)
    ax3.legend(frameon=True, facecolor="#1A1F2E", edgecolor="#2A2E3D", fontsize=8.5, loc="lower left")

    fig.suptitle("quantal response equilibrium (qre) homotopy continuation: newton-raphson logit tracing", fontsize=13.5, fontweight="bold", y=0.99)
    out = "qre_homotopy_simplex.png"
    plt.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"saved {out}")


if __name__ == "__main__":
    plot_qre_homotopy()
