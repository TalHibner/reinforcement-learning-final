# Academic Improvement Blueprint — from current submission to High Distinction

Companion to `GAP_ANALYSIS.md`. That document says what is broken; this one says what an
examiner would need to see to award the top band, and in what order to build it. No code here
by design — each item is a checklist entry with the governing mathematics, the acceptance
criterion, and the spec/rubric line it serves.

A note on "guarantee": Assignment3 §6 grades the *quality* of work beyond the requirements,
not its presence. Nothing guarantees a perfect score. What this blueprint does is (a) close
every stated requirement, (b) replace every ad-hoc design choice with a theorem- or
measurement-backed one, and (c) add two distinction-grade extensions chosen for insight per
unit of compute. That is the honest maximum.

> **Revision (2026-07-29), after lecturer clarifications.** Three rulings change this plan:
> (1) SimpleRoomEnv need not be *solved* — learning-process graphs suffice, though the
> lecturer still expects DQN/PPO to clear it "relatively quickly" → the Phase-1 gate becomes a
> diagnostic, not a compliance requirement (§1.5 below).
> (2) Two graphs are explicitly requested: **discounted reward vs episode** and **num steps vs
> episode** → new Phase-0 item 0.7 and reporting item 3.7; steps-per-episode already exists.
> (3) Submission = organized code files **plus a display-only Colab notebook that loads saved
> logs/outputs**, and the **report is the primary deliverable** → new Phase-0 item 0.8; the
> "Reload from Checkpoints" section moves from delete-or-implement to *implement*; report
> quality (Phase 5) outranks notebook re-runs in the priority matrix.

The three pillars, in grading order:

1. **Correctness floor** — nothing below counts while required deliverables are broken.
2. **Theory-grounded design** — every choice (shaping, bootstrapping, architecture,
   exploration) derivable from a stated principle, not a tuned accident.
3. **Statistical & diagnostic rigor** — multi-seed claims with uncertainty, and figures that
   *explain* behavior rather than just report it.

---

## Phase 0 — Restore the deliverable floor (prerequisite)

All from `GAP_ANALYSIS.md`; listed here only as gates. Do these before anything else.

- [ ] **0.1** Fix `_merge_logger` → correct cumulative-steps curve + JSON for DQN/ComplexEnv (C2).
- [ ] **0.2** Re-run the video cell; record + display all six partway clips; fix stale markdown (C3).
- [ ] **0.3** Add the dedicated best-settings cell (C4). Include: per-(algo, env) training
      hyperparameters, inference mode, model-selection rule, seed protocol.
- [ ] **0.4** Remove all claimed-but-absent artifacts or implement them (C6). The value
      baseline becomes real via §2.3. The "Reload from Checkpoints" section is **now
      mandatory** — it is exactly the lecturer's requested submission workflow (see 0.8).
- [ ] **0.5** Scrub advisory comments, dead code, malformed table row, duplicate imports (C7).
- [ ] **0.6** Storage-root: one definition, working fallback off-Colab (E5).
- [ ] **0.7 [lecturer #2] Discounted-return logging and plots.** Extend `TrainingLogger` to
      accumulate G₀ = Σ_t γ^t·r_{t+1} online per episode (one scalar each for shaped and
      sparse; store the γ used with the log). Add "discounted return vs episode" to the
      standard 4-panel figure (making it 6 panels with 3.7's speed curve, or a second row).
      For the runs already on disk, reconstruct the sparse version without retraining:
      the sparse reward fires only at the terminal goal step, so
      G₀^sparse = success · γ^(steps−1) — both factors are in the saved logs. State on the
      figure that shaped discounted returns exist only for post-fix runs.
- [ ] **0.8 [lecturer #3] Submission restructure.** (a) Factor the code into organized files
      (envs/wrappers, agents, training, evaluation, plotting) imported by the notebooks;
      (b) keep a *training* notebook (or scripts) that produces `artifacts/` — logs,
      checkpoints, figures, videos; (c) build the **display-only Colab notebook** that loads
      `artifacts/` and renders every required output with zero training — this is the
      "Reload from Checkpoints" section, implemented for real, as the deliverable notebook;
      (d) treat the report as the primary artifact: every figure it needs is exported by (b)
      at print quality.

**Acceptance:** clean top-to-bottom Run All *of the display notebook* on a fresh runtime in
minutes; every figure/video renders from saved artifacts; no cell output contradicts its
source; training notebooks/scripts documented as the producers of `artifacts/`.

---

## Phase 1 — Algorithmic correctness, with the mathematics made explicit

### 1.1 Time-limit bootstrapping — one policy for all three algorithms
The MDP objective is the infinite-horizon discounted return; `max_steps` is a *data-collection
artifact*, not environment dynamics (Pardo et al., 2018). Termination and truncation must flow
through the estimators differently:

- Terminal (goal / lava): target uses V(s') = 0.
- Truncated: target bootstraps through the cut.

Per algorithm:
- [ ] **PPO/GAE**: store `terminated` (not `terminated or truncated`) as the mask. For a step
      truncated at T: δ_T = r_T + γ·V(s_{T+1}) − V(s_T), not δ_T = r_T − V(s_T). The rollout
      already computes V(s_{T+1}) for the boundary bootstrap — route it to truncations too.
- [ ] **REINFORCE**: pure MC returns cannot bootstrap; two options, choose and *state* one:
      (a) with the learned baseline of §2.3, append γ^(T−t)·V̂(s_{T+1}) to truncated returns;
      (b) keep finite-horizon returns and declare the bias explicitly
      (|bias| ≤ γ^(T−t)·‖V‖∞ per step — negligible for SimpleRoom's T=100 successes, material
      for ComplexEnv's T=400 truncations).
- [ ] **DQN**: already correct — say so in the report and present the three-way treatment as a
      deliberate, unified design (this is exactly the §2.4 "discussion" rubric line).

**Acceptance:** a 5-line unit-test cell: hand-computed targets on a 3-state synthetic chain
with one truncation and one termination match the implementation for all three estimators.

### 1.2 DQN upgrade set (value-based family, done properly)
- [ ] **Double DQN.** Replace y = r + γ·(1−d)·max_a Q̄(s′,a) with
      y = r + γ·(1−d)·Q̄(s′, argmax_a Q(s′,a)).
      Motivation to state: for any random Q̂, E[max_a Q̂(s′,a)] ≥ max_a E[Q̂(s′,a)]
      (Jensen on the max), so vanilla DQN's target carries positive bias that compounds
      through bootstrapping; decoupling argmax (online) from evaluation (target) removes it
      (van Hasselt et al., 2016).
      **Distinction figure:** plot mean predicted Q(s₀, a₀) vs the *realized* discounted
      return from greedy eval rollouts, DQN vs DDQN — the overestimation gap made visible is
      textbook understanding demonstrated on your own data.
- [ ] **Dueling head.** Q(s,a) = V(s) + A(s,a) − (1/|A|)·Σ_{a′} A(s,a′) (the mean-subtraction
      resolves the V/A identifiability degeneracy). Task-specific justification to write:
      in ComplexEnv most states are navigation states where actions barely differ in value
      but *stage progress* changes V(s) enormously — precisely the regime the dueling
      decomposition was designed for (Wang et al., 2016).
- [ ] **n-step targets (n≈3).** y = Σ_{k=0}^{n−1} γ^k r_{t+k} + γ^n·(1−d)·Q̄(...).
      Rationale: milestone rewards propagate backward n cells per update instead of one —
      directly attacks the long prerequisite chain. State the caveat: uncorrected n-step from
      a replay buffer is off-policy-biased; justify n small.
- [ ] **Target sync decoupled from `train_freq`** (gap M4) + **learning-starts warmup**
      (~5–10k uniform-random steps before the first gradient step, so early Q-targets aren't
      fit to a 32-sample buffer).
- [ ] **ε schedule with eyes open** (gap M5): floor 0.05 (not 0.15); schedule length ≤ actual
      run length; and add a *greedy probe*: every 50 episodes, 5 ε=0 episodes logged
      separately. Training-metric and eval-metric now measure the same policy — this
      dissolves the "SR100=0.33 vs greedy 0%" mystery into a measured curve.
- [ ] Optional (stretch): **prioritized replay**. P(i) ∝ (|δ_i|+ε)^α with IS weights
      w_i = (N·P(i))^(−β)/max_j w_j, β annealed → 1. Justify from your own data: successful
      goal transitions are ~10⁻⁴ of the buffer; uniform sampling starves the only informative
      TD errors. Only take this if Phase 3 is already funded — it is the most engineering for
      the least marginal insight.

### 1.3 REINFORCE, made publication-clean
- [ ] **State the estimator you actually use.** The discounted-objective gradient is
      ∇J = E[Σ_t γ^t·∇log π(a_t|s_t)·(G_t − b(s_t))]; standard practice drops the γ^t factor.
      Say which you use and why (Sutton & Barto §13.3); silently mixing them is the kind of
      thing examiners probe in vivas.
- [ ] **Replace per-episode return whitening with a real baseline.** Whitening
      (G−μ_ep)/σ_ep is *not* a baseline: μ_ep depends on the episode's own actions (bias),
      and on all-failure episodes (constant −0.01 rewards) it manufactures full-magnitude
      gradients out of pure discount geometry (gap M2). Fix:
      (a) learned V̂(s) baseline (the §0.4 ablation, done right), trained on MC returns with
      an MSE head; variance-reduction argument to include:
      E[∇log π(a|s)·b(s)] = Σ_a ∇π(a|s)·b(s) = b(s)·∇Σ_a π(a|s) = b(s)·∇1 = 0,
      so any state-only baseline is unbiased, and b*(s) ≈ V^π(s) is near-optimal for variance.
      (b) If whitening is kept for comparison, whiten across a *batch* of k episodes
      (k≈8–16) — batching alone cuts gradient variance ~1/k.
- [ ] **Consistent loss scales** (gap M2): mean over steps for both the policy and entropy
      terms; entropy coefficient sweep {0.003, 0.01, 0.03} now that it does something.
- [ ] **Honest variance narrative**: quantify, don't gesture — report per-update gradient-norm
      variance REINFORCE vs PPO on the same env; one small figure, closes the §2.4
      variance-discussion rubric line with data instead of adjectives.

### 1.4 PPO to the "implementation matters" standard
(Engstrom et al., 2020; Huang et al., 2022 — cite both; examiners know this literature.)
- [ ] Truncation bootstrap (1.1) — the single highest-impact fix.
- [ ] **Shuffled minibatches**: rollout 1024 → 4–8 minibatches × 3–4 epochs; advantages
      normalized *per minibatch* ((Â−μ)/(σ+1e−8)).
- [ ] **Orthogonal initialization**: trunk gain √2; policy head gain 0.01 — near-zero logits
      ⇒ initial policy ≈ uniform ⇒ initial entropy ≈ ln|A| (maximal exploration by
      construction, not by luck); value head gain 1.0.
- [ ] **KL control done right**: use the low-variance unbiased estimator
      k₃ = (r−1) − ln r, r = π_new/π_old (Schulman's estimator note), computed *before*
      applying the epoch's step, early-stop at target_kl ≈ 0.015. The current code applies one
      update beyond the threshold every time (gap M3).
- [ ] **Entropy coefficient ≥ 0.01** on SimpleRoom (measured collapse to 0.428 with a wrong
      argmax is the failure signature — connect the fix to the measurement); linear LR anneal
      to 0.
- [ ] **Per-update diagnostics logged and plotted**: clip fraction, approx-KL, entropy,
      explained variance EV = 1 − Var(G−V)/Var(G). Four small curves that (i) catch collapse
      early and (ii) demonstrate you know what healthy PPO telemetry looks like.
- [ ] **Value-loss clipping: decide and justify.** Evidence says it often hurts (Engstrom
      et al.); either omission-with-citation or inclusion-with-ablation is fine — the
      *justification* is the rubric point.

**Phase-1 acceptance (revised per lecturer #1):** SimpleRoomEnv convergence is no longer a
compliance gate — learning-process graphs suffice. Treat it instead as the *diagnostic* it was
designed to be: the lecturer still expects DQN and PPO to solve it "relatively quickly", so
hold those two to ≥ 0.9 greedy as an internal health check (it is cheap, and greedy inference
is still how §4 results are reported); REINFORCE is exempt from convergence — show its
learning curve, apply the §1.3 fixes, and report the improvement honestly. The bit-identical
greedy-collapse finding (Gap C1) must still be resolved or explained, because it is a symptom
of M1/M3, not a cosmetic shortfall.

---

## Phase 2 — Theory-grounded design (the §6 extension material)

### 2.1 Observability proof → the frame-stacking answer (§2.2, currently unanswered)
Don't just say "no frame stacking" — *prove it is information-theoretically unnecessary*:
- [ ] Show the generator always places exactly 3 water balls and 3 lava tiles (water columns
      x ∈ {4,5,6} never intersect the reserved goal/lava cells — one paragraph from
      `_gen_grid`).
- [ ] Then every latent variable is a deterministic function of a single frame:
      carrying_key ⟺ key not visible; extinguished_count = 3 − #visible_lava;
      carrying_water ⟺ #visible_lava − #visible_balls = 1; door state is rendered directly;
      orientation is the triangle. Hence the pixel process is Markov and stacking adds cost,
      not information. **Decision: no stacking — with proof.**
- [ ] Honest caveat + measurement: decodable-in-principle ≠ decoded-by-the-network at 84×84 —
      which is what the linear probes in §2.4 test. (Proof + measurement together is
      distinction-grade treatment of one spec bullet.)

### 2.2 Potential-based reward shaping — one theorem replaces four hacks (centerpiece #1)
Current shaping provably changes the optimal policy: milestone mass (+13.5) exceeds the goal
(+10), and the trained PPO exploits exactly that optimum (milestones-then-idle; gap M8).
Replace the ad-hoc bonus zoo with shaping that *cannot* move the optimum:

- [ ] Define a stage potential over state-readable predicates (all of these are functions of
      the current state, not history — required for the theorem):
      Φ(s) = w₁·1[carrying_key] + w₂·1[door_open] + w₃·1[in_right_room]
           + w₄·1[carrying_water] + w₅·(#extinguished), with w monotone along the solution
      (e.g. 1, 3, 4, 5, +3 per tile).
- [ ] Shape with F(s, s′) = γ·Φ(s′) − Φ(s). **Theorem (Ng, Harada & Russell, 1999):** the
      optimal policy of the shaped MDP equals that of the unshaped MDP, for any Φ. (Wiewiora
      2003: equivalent to Q-table initialization.) State the theorem, cite it, and show the
      telescoping identity Σ_t γ^t·F_t = γ^T·Φ(s_T) − Φ(s₀) in one line.
- [ ] Work out the corollaries — this is where the marks are, because each one deletes one of
      your current hacks:
      1. **Farming is impossible by construction**: pickup→drop toggles Φ symmetrically, net
         shaped reward ≤ 0 for any loop. Delete the `water_mode` machinery and
         `drop_water_penalty`, and say *why* they are now unnecessary.
      2. **The death penalty emerges instead of being tuned**: at terminal states define
         Φ(terminal) = 0, so lava death yields F = −Φ(s) — death claws back exactly the
         accumulated potential. Dying late (more progress) costs more than dying early —
         the correct incentive, derived not chosen. Delete the ad-hoc −2.
      3. **Idling is punished by the telescope**: an agent that latches milestones then idles
         to truncation keeps only γ^T-discounted potential; the shaped return of
         milestones-then-idle is strictly less than milestones-then-goal. Your PPO's current
         pathology becomes suboptimal *by theorem*.
- [ ] Decide the step penalty separately: under PBRS, time preference is already carried by
      γ; if you keep −c per step, present it as a deliberate objective modification, not
      shaping.
- [ ] **The ablation that anchors the thesis**: {sparse only, current ad-hoc shaping, PBRS} ×
      {DQN, PPO} on ComplexEnv, one table + stage funnels. Even if PBRS does not fully solve
      ComplexEnv, "theorem-backed shaping removed the reward-hacking optimum our ad-hoc
      shaping created, and here is the behavioral evidence" is a distinction-grade result.

### 2.3 Exploration, diagnosed before treated
- [ ] **Quantify the bottleneck first.** Your funnel data localizes the break: door is
      reached, water is touched, lava extinguishing never happens (0 extinguish events in
      all 250 eval episodes). Estimate the probability of the bottleneck event under the
      current policy: it requires (carrying water) ∧ (adjacent to lava) ∧ (facing it) ∧
      (selecting `toggle`), i.e. roughly (1/|A|) × a small geometric occupancy factor —
      a back-of-envelope 10⁻³–10⁻⁴ per step. Presenting this estimate *is* the deep
      understanding; it also justifies the choice of remedy.
- [ ] **Primary remedies (cheap, legal, sufficient):**
      (a) ε floor 0.05 with the greedy probes of §1.2;
      (b) **curriculum via `max_steps`** — the only constructor knob the rules allow: train
      at 800 first (random walks need the room to complete the chain once), anneal to 400;
      report both budgets on the cumulative-steps axis so the comparison stays fair.
- [ ] **Stretch remedy (only if Phase 3 is funded): RND** (Burda et al., 2018). Intrinsic
      bonus r^i_t = ‖f̂_θ(o_t) − f(o_t)‖² for a fixed random target f; add with small
      coefficient, anneal; discuss non-stationarity of the combined reward and why
      novelty ≈ stage progress in this env (new stages expose never-seen pixel
      configurations). High insight, real engineering cost — take only as the second
      extension if time allows, and prefer depth on 2.2 otherwise (§6 rewards quality over
      count).

### 2.4 Representation probes — the "genuine understanding" figure set (centerpiece #2)
All cheap; all reuse trained networks; together they answer "what did the CNN actually learn?"
- [ ] **Linear probes**: freeze each trained trunk; fit logistic heads from the 512-d features
      to the ground-truth predicates {carrying_key, door_open, carrying_water,
      #lava remaining} (labels from the getters — analysis use, fully legal). Report probe
      accuracy per algorithm per training stage. DQN-vs-PPO probe accuracy differences give
      you a *mechanistic* explanation for the performance gap, not a speculative one.
- [ ] **Saliency at decision points**: |∂ max_a Q / ∂ pixels| (or logit gradient for PPO) at
      four canonical states (facing key; facing locked door with key; facing lava with water;
      facing goal). One 4-panel figure per algorithm.
- [ ] **Value heatmap**: for a fixed layout, V(s) (or max_a Q) at every free cell × 4
      orientations, before vs after the door opens. Uses read-only state placement for
      analysis only (eval-side, documented as such). This figure usually explains
      stalling behavior at a glance (flat value plateaus = no gradient to follow).

---

## Phase 3 — Experimental & statistical rigor (what makes it a thesis)

- [ ] **3.1 Multi-seed protocol.** N ≥ 3 seeds (5 for SimpleRoom, where runs are ~10 min) per
      (algorithm, env). Report **IQM** with 95% stratified bootstrap CIs, not means of one run
      (Agarwal et al., 2021 — cite it; single-run deep-RL claims are the field's
      known credibility hole, and examiners know it).
- [ ] **3.2 Paired comparisons.** You already evaluate all algorithms on identical seed sets —
      formalize it: paired design, McNemar's test on success indicators for each algorithm
      pair, exact binomial CIs (Wilson) on success rates. Report "PPO > REINFORCE, p = …"
      instead of adjacent bar heights.
- [ ] **3.3 Sample efficiency as the primary axis.** Headline curves plotted against
      cumulative env steps (fixed in 0.1), plus a **steps-to-criterion** table (env steps
      until greedy-probe SR ≥ 0.9 on SimpleRoom; censored entries marked ">budget"). This is
      §2.4's "efficiency matters" and §6's "small competition" line, answered directly.
- [ ] **3.4 Stage funnel with conditional probabilities.** Report P(stage k+1 | stage k) per
      algorithm per mode. Your existing eval data already yields e.g. PPO-greedy
      P(water | door) = 0/31 — the chain break localized in one number. Present the funnel for
      training and evaluation side by side.
- [ ] **3.5 Train/eval gap decomposition.** The current Discussion hand-waves the
      "SR100 0.33 vs greedy 0%" gap. Measure it: same checkpoint, 2×2 evaluation —
      {train-seed layouts, fresh layouts} × {ε = 0.15, ε = 0}. The gap factors into an
      exploration component and a generalization component, each with a CI. One table,
      replaces a paragraph of speculation with a measurement.
- [ ] **3.6 Reporting discipline.** Language rules for the final text: "consistent with", not
      "confirms", for n < 5 events; every number in the Discussion traceable to a cell output;
      stage summary tables report distribution means/medians, with best-case (max) clearly
      labeled if kept.
- [ ] **3.7 [lecturer #2] Speed-of-success diagnostics.** The lecturer's exact concern:
      success rate can plateau while the policy keeps improving *time-to-goal on the episodes
      it does win*. Two curves answer it, both derivable from existing logs:
      (a) **discounted return vs episode** (0.7) — on sparse rewards
      G₀ = γ^(T_goal−1) on successes, a strictly monotone transform of time-to-goal, so this
      curve rises exactly when the agent wins faster;
      (b) **success-conditioned episode length** — rolling mean of `steps` over successful
      episodes only, plotted beside the unconditional steps curve. Plot both per episode *and*
      against cumulative env steps (the lecturer reasons in env steps — "stuck after
      50,000 steps"). Apply to every algorithm × env; on ComplexEnv the same treatment applies
      to stage-milestones (time-to-door, time-to-water) even while goal successes are zero —
      that is the partial-progress version of the same insight.

---

## Phase 4 — Architecture & preprocessing refinements (small, each with its reason)

- [ ] **4.1 Integer-factor resize: 320 → 80, not 84.** 320/84 is fractional — tile boundaries
      alias across pixel rows and the same grid cell renders differently by position; 320/80
      = 4 exactly, and INTER_AREA becomes exact 4×4 block averaging (8 px per tile,
      artifact-free). One-line change, principled paragraph.
- [ ] **4.2 (Optional ablation) grid-aligned first conv.** With 80×80 inputs, an 8×8 kernel
      at stride 8 yields exactly one spatial unit per grid tile (10×10 feature map) — a
      task-informed alternative to the Atari-tuned Nature stack; compare parameter counts and
      SimpleRoom learning curves. Small, elegant, and it shows architecture chosen from task
      structure rather than cargo-culted.
- [ ] **4.3 Color justification made quantitative.** Keep RGB for ComplexEnv, but justify with
      the luminance table: gray = 0.299R + 0.587G + 0.114B maps key-purple and water-blue to
      nearby luminances while color separates them trivially; grayscale is retained for
      SimpleRoom where color carries no task state. (Turns an assumption into an argument.)
- [ ] **4.4 Robustness nit from E4**: `np.round` before the uint8 cast in the buffer, so
      storage stays exact under any future preprocessing change.
- [ ] **4.5 Compute accounting table.** Parameters per network, gradient steps per env step
      (replay ratio; PPO epochs×minibatches), wall-clock per run. Feeds the efficiency
      narrative of 3.3 and the reproducibility appendix.

---

## Phase 5 — The written thesis layer

- [ ] **5.1 Restructure the Discussion** into: Problem formalization (MDP, observability proof
      from 2.1, corrected state-space estimate — 2³ lava subsets, water-ball subsets included)
      → Methods (each algorithm's estimator with its equations and the truncation policy from
      1.1) → Results (multi-seed, CIs, funnels) → Analysis (probes, saliency, value maps, gap
      decomposition) → Limitations (what remains unsolved on ComplexEnv and *why*, with the
      bottleneck estimate from 2.3) → Reproducibility (seeds, versions, determinism caveats,
      compute).
- [ ] **5.2 References done properly**: Williams 1992 (REINFORCE); Sutton & Barto 2018;
      Mnih et al. 2015 (DQN); van Hasselt et al. 2016 (DDQN); Wang et al. 2016 (dueling);
      Schaul et al. 2016 (PER, if taken); Schulman et al. 2015 (GAE) & 2017 (PPO);
      Ng, Harada & Russell 1999 (PBRS); Wiewiora 2003; Pardo et al. 2018 (time limits);
      Burda et al. 2018 (RND, if taken); Engstrom et al. 2020; Huang et al. 2022;
      Agarwal et al. 2021 (statistics).
- [ ] **5.3 Choose TWO extensions and go deep** (§6 explicitly grades quality over presence).
      Recommended pair: **PBRS with its ablation (2.2)** + **statistical rigor with the gap
      decomposition (3.1–3.5)**. The probe suite (2.4) is the alternate if one of these
      stalls; RND and PER are stretch goals, not commitments.

---

## Priority matrix

| # | Item | Effort | Grade leverage | Depends on |
|---|------|--------|----------------|------------|
| 0.* | Deliverable floor | S | Unlocks everything | — |
| 0.7 | Discounted-return log + plot | S | **High (explicit lecturer request)** | — |
| 0.8 | Code files + display-only notebook | S–M | **High (explicit lecturer request)** | artifacts exist |
| 3.7 | Speed-of-success curves | S | High (lecturer's stated diagnostic) | 0.7 |
| 1.1 | Truncation bootstrapping | S | High | — |
| 1.4 | PPO overhaul | M | High | 1.1 |
| 1.3 | REINFORCE baseline + scales | S–M | High | — |
| 1.2 | DDQN + dueling + n-step | M | Medium-high | — |
| 2.2 | PBRS + ablation | M | **Highest (extension)** | 1.1, Phase-1 gate |
| 3.1–3.5 | Multi-seed + stats + funnel + gap decomposition | M | **Highest (rigor)** | Phase 1 |
| 2.1 | Observability proof | S | Medium-high | — |
| 2.4 | Probes + saliency + value maps | M | High | trained agents |
| 4.1/4.3 | Resize + color justification | S | Medium | — |
| 2.3(b) | max_steps curriculum | S | Medium | — |
| RND / PER | Stretch | L | Medium | everything above |

**Compute plan (Colab-realistic, from your own timings, revised per lecturer #1):**
SimpleRoom runs are 8–20 min each → 3 seeds × 3 algorithms ≈ 2 h is now sufficient (no
convergence chase for REINFORCE; DQN/PPO should clear the internal ≥0.9 check within existing
budgets once Phase 1 lands). Redirect the savings to ComplexEnv: DQN and PPO ≈ 1.5–2 h per
run → 3 seeds × 2 algos × 2 shaping conditions is the budget ceiling; run REINFORCE-Complex
at 1–2 seeds with a pre-registered stopping rule stated in your own words (a principled
budget decision, unlike the current "symbolic run" comment). Total ≈ 2–3 Colab days. If
constrained, cut PER/RND and the 4.2 ablation first — never the statistics, and never the
report (lecturer #3: the report is the primary deliverable).

## Definition of done

1. The display-only notebook Runs All clean in minutes from saved artifacts; every required
   output renders; no stale outputs; code files organized and imported (lecturer #3).
2. DQN and PPO clear the internal SimpleRoom check (≥0.9 greedy) with multi-seed CIs;
   REINFORCE shows a clear learning process with the §1.3 fixes applied (lecturer #1).
2a. Discounted-return and steps-per-episode curves present for every algorithm × env, plus
   the success-conditioned speed curves (lecturer #2).
3. Every estimator's equations appear once, with the truncation policy stated.
4. Shaping is PBRS with the invariance theorem cited, the three corollaries derived, and the
   three-way ablation reported.
5. Every headline claim carries a CI or a test; every figure is referenced from the text.
6. The frame-stacking question is answered with a proof and a probe measurement.
7. One table states best settings; one appendix states full reproducibility details.
8. The Limitations section quantifies why ComplexEnv remains hard (bottleneck-event estimate),
   rather than apologizing for it.
