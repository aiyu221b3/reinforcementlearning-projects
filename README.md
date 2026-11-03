# reinforcementlearning-projects

This repo is my workbench for reinforcement learning and game theory, with everything built from scratch! Every result gets plotted in dark mode, hehe.

It’s a work in progress, and it comes in two forms:

1. Tutorials: multi-file deep dives, each in its own folder.
2. Single-file projects: self-contained scripts in the root. Each one takes a single idea, rebuilds it from first principles, and then checks that it actually holds.

---

# Gallery

### 1. Recursive Belief-Based Learning (ReBeL) & Public Belief States
*Hidden cards turn into a smooth surface you can walk around on, and the kink lands exactly where the math says it should.*
![fig_1](images/rebel_pubbel_cfr.png)

Converts an imperfect-information extensive-form game into a continuous-state perfect-information game over the **Public Belief State (PBS)** simplex. Conditioning subgame solving on public posterior reach probabilities $\beta(c_i \mid h_{\text{pub}})$ maps out the exact piecewise-linear convex counterfactual value surface, verifying the indifference kink at $\beta^\star(\text{J} \mid \text{bet}) = \frac{1}{4}$ and the exact minimax root value $v^\star = -\frac{1}{18}$.

$$
V_{i}(c_{i} \mid \beta_{\text{pub}}) = \sum_{c_{-i}} P(c_{-i} \mid c_{i}, \beta_{\text{pub}}) \, u_{i}\left(\pi_{\text{sub}}^{\star}(\beta_{\text{pub}}); c_{i}, c_{-i}\right)
$$

---

### 2. Quantal Response Equilibrium (QRE) Homotopy & Logit Tracing
*Turn the rationality dial from “pure noise” to “perfect play” and watch the equilibrium trace out a path.*
![fig_2](images/homotopy_logit_trac.png)

Traces the smooth Quantal Response Equilibrium continuum from the high-entropy uniform centroid ($\tau \to \infty$) down to the unique **Limiting Logit Equilibrium** ($\tau \to 0^+$) on the 2-simplex via damped Newton-Raphson predictor-corrector continuation. Verifies the theoretical exploitability scaling bound $\text{NashConv}(\pi^*(\tau)) = \mathcal{O}(\tau \ln |\mathcal{A}|)$ along the entire logit branch.

$$
\pi_{i}^{\star}(a \mid \tau) = \frac{\exp\left(Q_{i}^{\pi^{\star}}(a) / \tau\right)}{\sum_{a' \in \mathcal{A}_{i}} \exp\left(Q_{i}^{\pi^{\star}}(a') / \tau\right)}
$$

---

### 3. Differentiable Games: LOLA & Stable Opponent Shaping (SOS)
*Naive learners sprint into mutual defection. Learners that model each other’s learning find Tit-for-Tat.*
![fig_3](images/sos_social_dil.png)

Solves the exact 5-state Markov representation of the Iterated Prisoner's Dilemma (IPD) analytically via the resolvent $\mathbf{M} = (\mathbf{I} - \gamma \mathbf{P})^{-1}$—zero finite-difference hacks or rollout variance. While naive gradient ascent marches straight into the mutual defection sink (`ALLD`, $V = -2.0$), differentiating through the opponent's learning step via **LOLA** and **SOS** bends the vector field into the reciprocal **Tit-for-Tat (`TFT`)** attractor ($V \to -1.0$).

$$
\Delta \theta_1^{\text{LOLA}} = \nabla_{\theta_1} V_1 + \delta \left( \nabla_{\theta_1 \theta_2}^2 V_2 \right)^\top \nabla_{\theta_2} V_1
$$

---

### 4. POMDP & Distributionally Robust Optimal Stopping (`finhor.py`)
*When do you stop if you can’t see the state, and your model of the world might be wrong too?*
![fig_4](images/finhor_stopping_analysis.png)

Solves a 7-state finite-horizon stopping corridor across **Sondik's Exact POMDP $`\alpha`$-Vector Pruning** (piecewise-linear convex upper envelope over the belief simplex $`\Delta^{6}`$), a **Dynkin Zero-Sum Stopping Game**, and **Nilim & El Ghaoui's KL-Divergence Distributionally Robust Bellman Operator** via Lagrangian convex duality over $`D_{\text{KL}}(Q \,\|\, P_{s,a}) \le \kappa`$.

$$
V_{k}^{\text{rob}}(s) = \max_{a \in \mathcal{A}} \left\lbrace r(s, a) + \gamma \sup_{\tau \ge 0} \left[ -\tau \kappa - \tau \ln \sum_{s'} P(s' \mid s, a) \exp\left(-\frac{V_{k-1}^{\text{rob}}(s')}{\tau}\right) \right] \right\rbrace
$$

---

### 5. Partially Observed Artifact Search & Self-Play Q-Learning (`marl_search_game.py`)
*Two agents, one beacon, and partial information on both sides. The pursuit and evasion are learned entirely through self-play.*
![fig_5](images/marl_search_game_analysis.png)

Trains a two-agent pursuit–evasion search game in a voxel-style obstructed arena. The **artifact runner** learns to recover the beacon while the partially informed **signal hunter** learns to intercept using only its position and the runner’s last observation inside a local sensing radius. The single dark-mode phase portrait combines self-play terminal frequencies, greedy-policy occupancy heatmaps, and early-versus-late policy compression, while verifying that recovery, interception, and timeout probabilities sum to $`100\%`$ in every reporting window.

$$
Q_i(o_t^i, a_t^i)
\leftarrow
(1 - \alpha) Q_i(o_t^i, a_t^i)
+
\alpha
\left[
r_t^i
+
\gamma (1 - d_t)
\max_{a' \in \mathcal{A}_i}
Q_i(o_{t+1}^i, a')
\right].
$$
