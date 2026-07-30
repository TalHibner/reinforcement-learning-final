# Discussion

## MDP Characterisation

Both environments are **episodic**, **fully observable** (the full grid is rendered in the image), and have **discrete action spaces**. From the agent's perspective, however, the observation is a high-dimensional image — the agent must infer latent task state (what it carries, whether the door is open, which lava tiles remain) purely from pixel colours, with no symbolic state available. That every latent variable *is* recoverable from one frame is the argument in the architecture section, and it is why no frame stacking is used.

**SimpleRoomEnv** has a manageable underlying state space: 8x8 = 64 interior cells x 4 facings x 4 goal corners ≈ **1,024 distinct configurations**. A Q-table would be feasible in principle, but the raw image (84 x 84 pixels per channel after preprocessing) makes tabular RL impractical and motivates function approximation.

**ComplexEnv** is vastly larger. Counting it in two factors, from the generator (`width = height = 10`, `partition_col = 3`, `n_lava = 3`, `n_water = 3`):

- **Layout multiplicity** (fixed within an episode, resampled every episode):
  16 key cells (x ∈ {1,2} × y ∈ {1..8}) × 8 door rows × 2 goal corners × 2 water corners = **512 layouts**.
- **Dynamic state within a layout:** 57 walkable cells (16 left + 40 right + the door cell) × 4 facings × 3 carry states (nothing / key / water) × 3 door states (locked, unlocked-closed, open) × 2³ = 8 lava-extinguished subsets × 2³ = 8 subsets of water balls still on the floor = **131,328**.

Product: **≈ 6.7 x 10⁷ configurations**, and this is an *upper bound* rather than a conservative estimate — the lava and water-ball subsets are correlated (each extinguish consumes one ball), so the reachable set is smaller. Either way a Q-table is completely impractical and deep function approximation is the only viable approach. (An earlier version of this estimate used 2⁴ = 16 lava subsets; the lava ring is 3 tiles in every layout — `len(lava_ring) == 3`, surfaced as `info["n_lava"]` — so the correct factor is 2³ = 8.)

**Reward sparsity and recoverable stages**: both environments deliver a sparse reward (+1 on goal, 0 otherwise). Within a ComplexEnv episode, dropping the key or spending a water ball without extinguishing lava are **recoverable mistakes** — the key remains on the grid, and other water balls stay available. Stepping onto unextinguished lava is **irrecoverable**: the episode terminates with reward 0. This asymmetry informed our shaping design: recoverable mistakes warrant only a mild penalty, while lava death warrants a strong one (−2).

---

## Corrections applied after the first full run

The results in this notebook come from a **second** run (Run B). A first run (Run A)
completed end-to-end and was then audited line by line against the specification;
the audit found five defects that materially affect the numbers, and all five are
fixed in the code above. They are listed here because the diagnosis is part of the
work, and because Run A's numbers are quoted below as the baseline the fixes are
measured against.

| # | Defect in Run A | Why it mattered | Fix |
|---|---|---|---|
| 1 | **PPO treated time-limit truncation as termination** (`done = terminated or truncated` in the GAE mask) | At every truncation the bootstrap was zeroed, telling the critic the world ends with value 0 at `max_steps`. On ComplexEnv essentially every episode truncates, so ~100% of episode boundaries injected a false terminal; the same pixel state at t=10 and t=390 then carries different targets — an irreducible noise floor in the value regression and therefore in every advantage (Pardo et al., 2018). | GAE bootstraps `γ·V(s_T)` on truncation and only cuts the trace; true terminals still use V = 0. The DQN loop was already correct; all three estimators now follow one policy, verified by a hand-computed unit test. |
| 2 | **REINFORCE mixed a per-episode SUM with a per-step MEAN** (`(logp*G).sum()` + `entropy.mean()`) | The policy term scaled with episode length T while entropy did not, so the effective entropy weight was ≈ coef/T ≈ 10⁻⁵–10⁻⁴ of the policy gradient: the advertised exploration bonus did nothing. Per-episode return whitening compounded this by manufacturing full-magnitude gradients on all-failure episodes out of pure discounting geometry. | Both terms are means over the episode's steps; the default baseline is running statistics across episodes, with per-episode whitening and a learned V(s) baseline available and **measured** in the ablation above. |
| 3 | **PPO ran full-batch updates, no minibatching, and its KL early-stop fired after the optimiser step** | Few full-batch steps per rollout is both less sample-efficient and more prone to the ratio saturation the clip is meant to manage; and one update beyond the KL threshold was always applied. | Shuffled minibatches with per-minibatch advantage normalisation, orthogonal init (policy-head gain 0.01), linear LR anneal, and the KL check (estimator `(r−1) − log r`) moved *before* the step. |
| 4 | **DQN target sync was coupled to `train_freq`** (the check lived inside `update()`) | Syncs happened only on steps divisible by both, so the effective period was `lcm(target_update, train_freq)`. The shipped configs survived by arithmetic luck; the sanity config (`target_update=10, train_freq=4`) silently synced at half the configured rate. | `maybe_sync_target()` runs on the environment-step clock, called once per env step. Regression test asserts 10 syncs in 100 steps at `target_update=10, train_freq=4`. |
| 5 | **Shaping mass exceeded the goal payoff** (+13.5 of milestones vs a goal of +10) | The shaped optimum plausibly excluded reaching the goal at all — and Run A's PPO found exactly that optimum: key → door → water, then idle to the 400-step cap, forever. | Shaping rebalanced so the *discounted* milestone total is strictly below the discounted goal payoff along a competent trajectory, asserted by the audit cell (ratio 0.56 vs Run A's 3.60). |

Three further corrections affect reporting rather than learning: the chunk-merged
DQN/ComplexEnv logger dropped `cumulative_steps`, so the required
cumulative-environment-steps curve spanned only the final 200 of 4,000 episodes
(making DQN look like it had used ~5% of PPO's sample budget when it had used
~85%); model selection was asymmetric (DQN evaluated at its best checkpoint, the
other two at final weights); and training success rates were measured under an
ε = 0.15 behaviour policy while the reported evaluation was greedy. All three are
fixed: the cumulative axis is rebuilt from the merged episode lengths, one
`BestCheckpointer` rule applies to all three algorithms, and periodic greedy probes
are plotted alongside the behaviour-policy curve.

**Run A results, for reference** (evaluation as reported then — 200 episodes on
SimpleRoomEnv, 50 on ComplexEnv):

| Algorithm | Mode | Success % | Avg Sparse | Avg Shaped | Avg Steps |
|---|---|---:|---:|---:|---:|
| DQN | greedy | 90.5 | 0.905 | +8.87 | 18.1 |
| REINFORCE | greedy | 5.0 | 0.050 | −0.45 | 95.2 |
| REINFORCE | sample | 46.0 | 0.460 | +3.86 | 74.3 |
| PPO | greedy | 5.0 | 0.050 | −0.45 | 95.2 |
| PPO | sample | 56.0 | 0.560 | +4.93 | 66.8 |

with ComplexEnv at 0% sparse success for all three algorithms (DQN reaching stage
Lava in 1 of 50 episodes, PPO-sampled reaching Water in 30 of 50). Two details in
that table are themselves diagnostic: the REINFORCE-greedy and PPO-greedy rows are
identical to printed precision across all four statistics over the same 200 seeds,
which is not plausible for two independently trained networks unless both argmax
policies had collapsed to the same near-constant behaviour (≈5% of random spawns
already face the goal line; failures truncate at 100 steps, hence ≈95 mean steps).
That collapse is the observable symptom of defects 2 and 3, which is why it is
reported as a bug rather than as a finding about greedy inference.

**Run B results are the auto-generated tables above** — they are rendered from this
run's evaluation dictionaries rather than typed in, so they cannot drift out of step
with the run that produced them.

---

## Algorithm Comparison

| Property | DQN | REINFORCE | PPO |
|---|---|---|---|
| **Family** | Value-based (off-policy) | Policy-gradient (on-policy) | Actor-Critic (on-policy) |
| **Sample efficiency** | Highest (experience replay) | Lowest (discards every transition) | Middle (rollout reused over epochs) |
| **Stability** | Target network + Huber loss | High variance (Monte Carlo returns) | Clipped surrogate bounds updates |
| **Exploration** | Explicit ε-greedy schedule + greedy probes | Stochastic policy + entropy bonus | Stochastic policy + entropy bonus |
| **Time-limit handling** | Bootstraps through truncation | Finite-horizon MC returns (bias stated; bootstrapped when the value baseline is on) | Bootstraps through truncation, trace cut |
| **Model selection** | Best-so-far by rolling SR100 | Best-so-far by rolling SR100 | Best-so-far by rolling SR100 |
| **Training budget (ComplexEnv)** | 4000 episodes | 4000 episodes (early-stop rule, patience 1000) | 4000 episodes |

Per-algorithm evaluation numbers are in the generated tables above; the comparison
figures place all three on a common environment-step axis.

### Evaluation Methodology — Greedy vs Sampled Modes

For every trained agent we report two evaluation modes:

- **Greedy**: argmax of the Q-values (DQN) or of the policy logits (REINFORCE, PPO).
  Greedy evaluation is deterministic given the trained weights and the
  environment reset seed.
- **Sampled**: for policy-gradient methods only, actions are sampled from the
  learned policy. Sampled evaluation depends on the PyTorch RNG state as well
  as the environment seed. To make sampled evaluation reproducible across
  reruns and independent of the number of PyTorch operations performed during
  training, the PyTorch RNG is reseeded per evaluation episode using a
  deterministic seed derived from the evaluation seed and the episode index.

Sampled evaluation is used for REINFORCE and PPO because those methods
optimise a stochastic policy during training. Reporting only greedy evaluation
would systematically underestimate their learned behaviour whenever the
argmax action is brittle. Reporting *only* sampled evaluation would be the
opposite error, so both are shown side by side, and a large gap between them is
treated as a diagnostic (a confident-but-wrong argmax) rather than as a result.

**Evaluation episode counts:**

- SimpleRoomEnv: `n_episodes=200` per algorithm and mode.
- ComplexEnv:    `n_episodes=50`  per algorithm and mode.

The larger SimpleRoomEnv count reduces sampled-evaluation granularity from
2 percentage points per successful episode (at 50 episodes) to 0.5 percentage
points (at 200 episodes). ComplexEnv is kept at 50 episodes given its
much higher per-episode cost — with the consequence, stated in the results table,
that any statistic worth less than 1/50 of an episode is an anecdote.

| Environment    | Evaluation modes   | n_episodes | PyTorch RNG per episode | Env reset seeded |
|----------------|--------------------|------------|-------------------------|------------------|
| SimpleRoomEnv  | greedy, sampled    | 200        | Yes (sampled only)      | Yes              |
| ComplexEnv     | greedy, sampled    | 50         | Yes (sampled only)      | Yes              |

---

## ComplexEnv Results

### Training configuration

All three algorithms get the same 4,000-episode budget and the same model-selection
rule. DQN uses a replay buffer of 100,000 transitions, batch size 64,
`train_freq=4`, target sync every 1,000 environment steps, and a linear ε decay to
**0.05** over 600,000 steps — Run A held ε at 0.15 for the whole run, which meant
its headline training success rate was measured under a 15%-random policy while the
reported evaluation was greedy. With the floor at 0.05 and greedy probes every 250
episodes, the training curve and the evaluation now describe the same policy, and
the probe curve is the honest answer to "is it learning?".

PPO uses 1,024-step rollouts with 4 shuffled minibatches over 4 epochs, entropy
coefficient 0.015, `target_kl=0.015` and an annealed learning rate. REINFORCE uses
the running baseline with entropy coefficient 0.02, plus the documented early-stop
rule; if it stops before 4,000 episodes, the stopping episode is printed by its
training cell and the cumulative-steps panel shows exactly how much of the budget
it consumed.

### Interpreting the gap between training success and greedy evaluation

This gap was the most interesting finding of Run A and it survives as a
methodological point: a rolling training success rate measured under an exploratory
behaviour policy is **not** an estimate of greedy performance, and on a
randomized-layout task it is not an estimate of held-out performance either.
Run A reported SR100 = 0.33 at episode 3,600 alongside 0% greedy success on 50
fresh layouts, and attributed the gap to three causes: 15% random actions during
training, per-episode layout randomization (so evaluation seeds are different maze
instances), and a deterministic argmax that gets stuck when it is confidently wrong.
Only the third of those is a property of the learned policy. Run B separates them by
construction — the ε floor is 0.05, the greedy probes measure the argmax policy
during training on held-out probe seeds, and best-checkpoint selection is driven by
the same metric for every algorithm — so any remaining gap is attributable rather
than mysterious.

### Reporting stage progress honestly

The stage columns report **mean and median** alongside the maximum. `MaxStage` is a
single best episode out of 50 and Run A's discussion led with it, which overstates
typical behaviour; the full distribution is printed in the stage breakdown and
plotted in the grouped bar chart. Similarly, an "Avg Ext." of 0.02 over 50 episodes
is **one lava tile extinguished, once** — consistent with a single lucky trajectory
and not evidence that the agent learned water-to-lava extinguishing. Run A's text
claimed the latter; it should not have.

---

## Reward Shaping Analysis

### The budget constraint, made explicit

Event shaping is safe only if the agent cannot do better by farming milestones than
by finishing the task, and the comparison that decides this is the **discounted**
one: milestones fire early, the goal fires ~200 steps later, and at γ = 0.99 the
goal is worth γ²⁰⁰ ≈ 0.13 of its face value at the start of an episode. The audit
cell computes both sides along an explicit competent trajectory and asserts

    sum_i γ^{t_i} · b_i  <  γ^{T_goal} · goal_scale

Run A's configuration (`goal_scale=10`, key 1, door 2, right-room 1, water 0.5,
lava 3/tile) fails this by **3.60x**: +13.5 of milestone mass against a goal worth
+10 undiscounted and +1.34 discounted. "Collect the milestones, then idle" was
genuinely the better shaped policy, and Run A's PPO converged to precisely that:
key → door → water for a net ≈ +0.1…+0.5, then 400 steps of step-penalty with
sparse return 0.00, every episode, for thousands of episodes. That is not a training
failure — it is the agent solving the problem we actually posed.

Run B's configuration (`goal_scale=20`, key 0.5, door 1.0, right-room 0.25,
water 0.2, lava 0.5/tile) passes at a ratio of **0.56**, and the total milestone
mass (+3.45) is now smaller than the 400-step penalty budget (−4.00), so
milestones-then-idle has a negative return and cannot be an optimum.

### What the shaping is for, and what it cannot fix

- `water_mode="first"` — reward the first water pickup once per episode, which
  removes the pickup/drop farming exploit that inflated Run A's shaped returns.
- The key bonus is the one milestone a replay buffer can amplify: DQN reaches
  stage Key in a large fraction of episodes in both runs.
- The lava-death penalty (−2) discourages walking into the ring; Run A's episode
  returns shifted from −4.0 toward −3.0 late in training, consistent with fewer
  lava deaths even without task success.
- **What it cannot overcome:** for the on-policy methods, milestone events are so
  rare early on that the shaped return is dominated by the step penalty alone
  (−0.01 × 400 = −4.0), which carries no directional signal. Shaping cannot
  manufacture a gradient towards an event that never occurs.

**Step penalty vs exploration trade-off**: the step penalty discourages wasted
steps but also penalises the deep backtracking stage 1 requires (navigate to key →
pick up → navigate to door → toggle → enter right room: easily 100–150 steps before
any positive reward). For long prerequisite chains, a smaller step penalty or a time
budget scaled to the chain length would allow more exploratory behaviour; the step
penalty here is kept because the shaping-budget argument above depends on it being
large enough to make idling unprofitable.

**Distinguishing sparse vs shaped returns**: the `SparseReturnTracker` wrapper sits
innermost and captures the environment's raw reward before any shaping wrapper
touches it, exposing it via `info['sparse_reward']`. Every training loop and every
evaluation table reports both metrics independently, and the training figures plot
both the undiscounted and the discounted version of each. This distinction is what
makes the water-farming diagnosis possible at all: an agent farming shaped reward
with sparse return 0.00 looks like it is learning until you plot the two separately.

---

## Key Design Choices

- **Artifact checkpointing.** All `TrainingLogger` metrics, matplotlib figures and
  model weights (final *and* best-so-far) are flushed to `artifacts/` under
  `STORAGE_ROOT` as training proceeds, so results survive a runtime disconnect.
  `STORAGE_ROOT` is resolved once (mounted Drive → Colab session storage → a local
  folder), so the notebook also runs off-Colab.
- **Reload from Checkpoints.** The section above implements this for real:
  `LOAD_FROM_ARTIFACTS = True` rebuilds all six agents and loggers from the saved
  artifacts, so the notebook can be run top-to-bottom in minutes with no training,
  and every figure, table and video regenerates from stored state. Which weights
  are loaded is explicit (`prefer_best=True` → the best-so-far checkpoint, the same
  weights the reported results use).
- **One source of truth for hyperparameters.** The dedicated best-settings cell
  defines `BEST_SETTINGS`, `EPISODE_BUDGET`, `INFERENCE_SETTINGS`,
  `MODEL_SELECTION` and `SEED_PROTOCOL`; the training cells pass
  `**BEST_SETTINGS[...]` rather than repeating numbers, so the stated settings are
  necessarily the settings that ran.
- **Regression tests, not just sanity checks.** Every assertion in the checks cell
  targets a defect that was actually present: hand-computed GAE targets for
  truncation vs termination, target-sync frequency under `train_freq=4`,
  episode-length invariance of the REINFORCE loss, replay-buffer next-state
  linkage, and repair of legacy logs.

The agent consumes only preprocessed pixel observations:

- SimpleRoomEnv: single-channel grayscale, 84x84, float32 in [0,1].
- ComplexEnv: 3-channel RGB, 84x84, float32 in [0,1] (colour is object identity here).
- **No frame stacking** — every latent variable is a function of the current frame;
  the argument, and the two caveats that go with it, are in the architecture section.

The action space is a reduced discrete subset of MiniGrid actions:

- SimpleRoomEnv: left, right, forward.
- ComplexEnv: left, right, forward, pickup, drop, toggle.

Reward shaping reads **event counts only** (`is_carrying_*`, `is_door_open`,
`extinguished_lava_count`, and `info["n_lava"]`); no position getter is consulted
inside the reward path, so no geometry can leak into the reward. Run A used
`lava_positions()` for a "is any lava left" test — a positions getter the
environment documents as being for analysis — which is now
`info["n_lava"] − extinguished_lava_count()`.

---

## Why PPO over A2C

The assignment requires at least one actor-critic method. We implemented **PPO directly** rather than starting with A2C for three reasons:

1. **Infrastructure validation does not require A2C.** PPO on SimpleRoomEnv validates the shared CNN backbone, actor-critic heads, GAE advantage computation, and rollout-based training loop just as effectively as A2C would.

2. **The incremental complexity is small.** The differences between A2C and PPO are the clipped surrogate objective and multiple optimisation epochs per rollout — approximately 15 lines of code and no new data structures.

3. **PPO is strictly more robust.** A2C applies unconstrained policy updates, which can cause large destructive steps — exactly the failure mode observed with REINFORCE. PPO's clip ratio (ε = 0.2) explicitly bounds the policy change per update, an important safeguard on ComplexEnv's long-horizon sparse-reward setting.

The Run A caveat is worth keeping: even PPO struggled on ComplexEnv, and the
audit found that part of that struggle was implementation (truncation
bootstrapping, minibatching, KL timing) rather than algorithm choice. Sample
efficiency — where DQN's replay buffer has a structural advantage — remains the
more plausible bottleneck than A2C vs PPO.

---

## Conclusion

The experiments separate cleanly into two regimes. SimpleRoomEnv is a pipeline test:
it does not have to be solved to convergence, but a policy-gradient agent whose
greedy success rate collapses to 5% on it is reporting a bug, not a finding — which
is what Run A's identical REINFORCE/PPO greedy rows turned out to be. ComplexEnv is
the task, and there the decisive result of Run A was not any single number but the
diagnosis: with a milestone budget larger than the discounted goal payoff, the
shaped optimum excluded the goal, and PPO found that optimum exactly. Fixing the
reward budget, the truncation bootstrapping, the loss scales and the PPO update
mechanics changes what the algorithms are being asked to optimise, which is why
every number is regenerated rather than patched.

What holds across both runs is the methodological core: sparse and shaped returns
reported separately at every stage, evaluation on a fresh seed space with greedy and
sampled modes both shown, one model-selection rule for all three algorithms, all
three placed on a common environment-step axis, and stage progress reported as a
distribution rather than a best case. Those are the parts that make the comparison
mean something — and they are also what made the defects above findable.
