# Discussion

## MDP Characterisation

Both environments are **episodic**, **fully observable** (the full grid is rendered in the image), and have **discrete action spaces**. From the agent's perspective, however, the observation is a high-dimensional image — the agent must infer latent task state (what it carries, whether the door is open, which lava tiles remain) purely from pixel colours, with no symbolic state available.

**SimpleRoomEnv** has a manageable underlying state space: ~64 agent positions × 4 facings × 4 goal corners ≈ **1,024 distinct configurations**. A Q-table would be feasible in principle, but the raw image (84 × 84 pixels per channel after preprocessing) makes tabular RL impractical and motivates function approximation.

**ComplexEnv** is vastly larger. A conservative product estimate: ~16 key positions × 8 door rows × 2 goal corners × 2 water corners × 3 carry states (none / key / water) × 2 door states × 2⁴ lava extinguishing combinations × ~64 walkable cells × 4 facings ≈ **12 million distinct configurations**. A Q-table is completely impractical; deep function approximation is the only viable approach.

**Reward sparsity and recoverable stages**: both environments deliver a sparse reward (+1 on goal, 0 otherwise). Within a ComplexEnv episode, dropping the key or spending a water ball without extinguishing lava are **recoverable mistakes** — the key remains on the grid, and other water balls stay available. Stepping onto unextinguished lava is **irrecoverable**: the episode terminates with reward 0. This asymmetry informed our shaping design: recoverable mistakes warrant only a mild penalty, while lava death warrants a strong one (−2).

---

## Algorithm Comparison

| Property | DQN | REINFORCE | PPO |
|---|---|---|---|
| **Family** | Value-based (off-policy) | Policy-gradient (on-policy) | Actor-Critic (on-policy) |
| **Sample efficiency** | Highest (experience replay) | Lowest (discards every transition) | Middle (single-pass rollout) |
| **Stability** | Target network + Huber loss | High variance (Monte Carlo returns) | Clipped surrogate bounds updates |
| **Exploration** | Explicit ε-greedy schedule | Stochastic policy + entropy bonus | Stochastic policy + entropy bonus |
| **SimpleRoomEnv eval SR** | **28%** | 2% | 2% |
| **ComplexEnv best eval stage** | Key (8 / 50 episodes) | None (0 / 50) | None (0 / 50) |

### Final Results — SimpleRoomEnv

| Algorithm  | Mode    | Success % | Avg Sparse | Avg Shaped | Avg Steps |
|---|---|---:|---:|---:|---:|
| DQN        | greedy  | 90.5 | 0.905 | +8.87  | 18.1 |
| REINFORCE  | greedy  | 5.0  | 0.050 | -0.45  | 95.2 |
| REINFORCE  | sample  | 46.0 | 0.460 | +3.86  | 74.3 |
| PPO        | greedy  | 5.0  | 0.050 | -0.45  | 95.2 |
| PPO        | sample  | 56.0 | 0.560 | +4.93  | 66.8 |

DQN solved SimpleRoomEnv reliably, reaching 90.5% success under greedy evaluation with an average episode length of 18.1 steps. Under greedy argmax evaluation, REINFORCE and PPO obtained only 5.0% success, but sampled-policy evaluation revealed meaningful learned stochastic behaviour: 46.0% for REINFORCE and 56.0% for PPO. This gap is consistent with the confidence diagnostics: PPO learned a confident but imperfect policy (mean max probability 0.849), and REINFORCE learned a less committed but non-uniform policy (mean max probability 0.629). The gap between greedy and sampled success indicates that the argmax action of these policies is often locally wrong, while sampling escapes those failure loops. Evaluation used 200 episodes per algorithm and mode, and PyTorch was reseeded per evaluation episode to make sampled evaluation reproducible.

### Evaluation Methodology

For every trained agent we report two evaluation modes:

- **Greedy**: argmax of the Q-values (DQN) or of the policy logits (REINFORCE, PPO). Greedy evaluation is deterministic given the trained weights and the environment reset seed.
- **Sampled**: for policy-gradient methods only, actions are sampled from the learned policy. To make sampled evaluation reproducible and independent of training-time PyTorch RNG usage, the PyTorch RNG is reseeded per evaluation episode using a deterministic seed derived from the evaluation seed and the episode index.

Sampled evaluation is used for REINFORCE and PPO because those methods optimise a stochastic policy during training. Reporting only greedy evaluation would systematically underestimate their learned behaviour whenever the argmax action is brittle. SimpleRoomEnv evaluation uses 200 episodes per algorithm and mode. ComplexEnv evaluation uses 50 episodes per algorithm and mode, given the greater per-episode cost.

| Algorithm | Final Train SR100 | Greedy Eval SR | Sampled Eval SR | Avg Greedy Steps |
---|---:|---:|---:|---:|
| DQN | 1.00 | 0.90 | N/A | 19.3 |
| REINFORCE | 0.54 | 0.02 | 0.60 | 98.1 |
| PPO | 0.63 | 0.02 | 0.56 | 98.1 |

### ComplexEnv Results

### ComplexEnv — DQN

We trained DQN on ComplexEnv for 4000 episodes using the anti-farming shaping
(`water_mode="first"`, `water_bonus=0.5`, `lava_bonus=3.0`), plus `right_room_bonus=1.0`
and `drop_water_penalty=0.2`. Exploration used a slow linear epsilon decay to a floor of
`eps_end=0.15` over 800,000 environment steps. The DQN used a replay buffer of 100,000
transitions, batch size 64, `train_freq=4`, and target network sync every 1,000 steps.

The training was chunked into 200-episode blocks. After each block, the rolling
success rate over the last 100 episodes (`SR100`) was computed, and the DQN weights
were saved as the best-so-far checkpoint whenever `SR100` improved. This protects
against DQN's known failure mode of overwriting a good policy late in training.

Under this configuration, DQN reached peak `SR100 = 0.33` at episode 3600. Individual
training episodes reached shaped returns above +14 and +16 after episode 2600, which
correspond to actual full-task solutions (key → door → water → extinguish → goal).
The exploration floor of 0.15 helped DQN keep discovering successful trajectories
even after episode 2400, when epsilon reached its minimum. Evaluation is performed
using the DQN weights from the checkpoint at episode 3600 rather than the
end-of-training model, ensuring we report the strongest observed DQN policy rather
than a degraded one.

### ComplexEnv — PPO and REINFORCE

PPO ComplexEnv training in this configuration was cut short by a Colab runtime timeout before evaluation could be run.

REINFORCE showed no rolling success on ComplexEnv, consistent with earlier runs. It remains the weakest method on this task, likely because of high variance in Monte Carlo policy gradients on long-horizon sparse-reward tasks.

---|---|---:|---:|---:|---:|---:|---:|
| DQN | Greedy | 2.0 | 0.02 | -2.60 | Goal | 0.0 | 0.04 |
| REINFORCE | Greedy | 0.0 | 0.00 | -3.98 | Key | 0.0 | 0.00 |
| REINFORCE | Sample | 0.0 | 0.00 | -3.15 | Door | 0.0 | 0.00 |
| PPO | Greedy | 0.0 | 0.00 | -3.88 | Key | 0.0 | 0.00 |
| PPO | Sample | 0.0 | 0.00 | -0.37 | Water | 4.0 | 0.00 |

PPO showed the strongest partial progress under sampled evaluation: it reached
Door in 25/50 episodes and Water in 24/50 episodes, but never reached the Lava or
Goal stages. This suggests PPO learned the early navigation and object-acquisition
subtasks but failed at the water-to-lava interaction. REINFORCE remained weakest,
although sampled evaluation reached the Key stage in most episodes and Door in a
small number.

| Algorithm | Mode | None | Key | Door | Water | Lava | Goal |
|---|---|---:|---:|---:|---:|---:|---:|
| DQN | Greedy | 26 | 14 | 7 | 1 | 1 | 1 |
| REINFORCE | Greedy | 49 | 1 | 0 | 0 | 0 | 0 |
| REINFORCE | Sample | 12 | 36 | 2 | 0 | 0 | 0 |
| PPO | Greedy | 44 | 6 | 0 | 0 | 0 | 0 |
| PPO | Sample | 0 | 1 | 25 | 24 | 0 | 0 |

**Policy-Gradient Performance**:
For policy-gradient agents, sampled evaluation was more informative than greedy argmax. On SimpleRoomEnv, both REINFORCE and PPO achieved meaningful sampled success despite poor greedy evaluation. On ComplexEnv, sampled evaluation also improved early-stage progress relative to greedy evaluation, but neither algorithm learned the water-lava subtask. PPO with lower learning rate, longer rollouts, entropy regularisation and target-KL early stopping improved stability but did not solve the core long-horizon dependency.

A learned state-value baseline was tested for REINFORCE. Although this should reduce variance in theory, in our implementation it degraded performance: the learned policy remained close to uniform on SimpleRoomEnv and achieved substantially lower sampled evaluation success than the simpler return-normalised REINFORCE variant. We therefore report the simpler REINFORCE variant as the main policy-gradient baseline and discuss the value-baseline version as an unsuccessful ablation.


**DQN** was the only algorithm to reach stage 1: it picked up the key in 8 of 50 greedy evaluation episodes (16%). During training, episode rewards ranged between −0.5 and −4.0 rather than flatly bottoming out at −4.0, consistent with the agent occasionally triggering the key-pickup bonus (+1.0) and avoiding some lava deaths (the lava-death penalty of −2.0 drives rewards below −4.0 at truncation). The replay buffer explains DQN's advantage: the rare transition in which the agent picks up the key is stored and replayed many times, gradually reinforcing that sub-policy even though the overall episode return remains negative.

**REINFORCE and PPO** made zero measurable progress on ComplexEnv: every evaluation episode ended at stage 0 (None: 50 / 50). For on-policy methods, a successful transition contributes to exactly one gradient update before being discarded. Given the low probability of reaching the key in a random 400-step episode, neither algorithm accumulated enough positive signal to bootstrap key-seeking behaviour. REINFORCE's high variance compounded the problem: the cumulative step penalty over 400 steps (−4.0) dominated the return in virtually every episode, producing near-zero gradients regardless of early-stage progress.

---

## Reward Shaping Analysis

The final ComplexEnv shaping used bounded event-based rewards:

- water_mode = "first" — reward the first water pickup once per episode,
  preventing repeated pickup/drop farming.
- water_bonus = 0.5 — modest bounded discovery reward for water.
- lava_bonus = 3.0 — stronger per-tile reward for actually extinguishing lava.

This shaping was chosen after an ablation across three variants:

- Original gated water pickup shaping, which was exploitable through repeated
  pickup/drop cycles.
- Strict per-lava water shaping, which removed the exploit but produced too
  sparse a signal after the door stage.
- The first-water compromise used here, which prevents farming while retaining
  a bounded discovery signal.

**What worked**: For DQN on ComplexEnv, the key-pickup bonus (+1) provided a learnable intermediate reward that the replay buffer could amplify across many replays. The lava-death penalty (−2) appears to have discouraged DQN from entering the right room recklessly — episode rewards shifted from −4.0 toward −3.0 in later training, consistent with fewer lava deaths even without solving the full task.

**What it could not overcome**: For on-policy methods, the shaping was ineffective because neither algorithm consistently reached any milestone. Without milestone events to reward, the shaped return was dominated by the step penalty alone (−0.01 × 400 = −4.0 per episode), providing no directional gradient signal above noise.

**Step penalty vs. exploration trade-off**: The step penalty discourages unnecessary steps but also penalises the deep backtracking that stage 1 requires (navigate to key → pick up → navigate to door → toggle → enter right room — easily 100–150 steps before any positive reward fires). For environments with long prerequisite chains and sparse milestone events, a smaller step penalty or a time-budget scaled explicitly to the chain length might allow more exploratory behaviour.

**Reward-Shaping Ablation**:
The original ComplexEnv shaping produced higher shaped returns and deeper partial progress, but DQN sometimes accumulated very large shaped rewards without sparse success or lava extinguishing. This suggested reward exploitation, most likely repeated water-pickup reward farming. We therefore introduced an anti-farming shaping variant that restricted water-pickup reward and increased the relative value of lava extinguishing.

After this change, the inflated shaped returns disappeared, confirming that the original reward was exploitable. However, progress also dropped: no method reached the lava stage or extinguished any lava tile. This indicates that the water-pickup reward had acted as both a useful exploration signal and a local optimum. The final bottleneck is the post-door object-manipulation chain: carry water, face lava, toggle, repeat, and then navigate to the goal.

**Distinguishing sparse vs. shaped returns**: The `SparseReturnTracker` wrapper captures the environment's raw reward before any shaping wrapper modifies it, exposing it via `info['sparse_reward']`. All training loops and the evaluation table report both metrics independently. This distinction matters: without it, an agent farming shaped rewards without ever reaching the goal would appear to be learning. The evaluation table confirms that no algorithm achieved non-zero sparse returns on ComplexEnv, and that the near-zero SimpleRoomEnv sparse returns for REINFORCE and PPO (2% each) are not artefacts of reward scaling.

---

## Key Design Choices

- **Artifact Checkpointing**: To prevent catastrophic data loss from runtime disconnects during long training sessions, we implemented a robust artifact tracking system. All `TrainingLogger` metrics, matplotlib figures, and final model weights (`save_agent`) are immediately flushed to an `artifacts/` directory on disk. This guarantees that training curves and evaluation statistics survive kernel restarts. Furthermore, we provided an optional **'Reload from Checkpoints'** section that allows anyone reviewing the notebook to instantiate the agents directly from these saved artifacts, skipping the lengthy training cells and enabling rapid evaluation and video generation.

The agent consumes only preprocessed pixel observations:

- SimpleRoomEnv: single-channel grayscale, 84x84, float32.
- ComplexEnv: 3-channel RGB, 84x84, float32.

The action space is a reduced discrete subset of MiniGrid actions:

- SimpleRoomEnv: left, right, forward.
- ComplexEnv: left, right, forward, pickup, drop, toggle.

---

## Why PPO over A2C

The assignment requires at least one actor-critic method. We implemented **PPO directly** rather than starting with A2C for three reasons:

1. **Infrastructure validation does not require A2C.** PPO on SimpleRoomEnv validates the shared CNN backbone, actor-critic heads, GAE advantage computation, and rollout-based training loop just as effectively as A2C would.

2. **The incremental complexity is small.** The differences between A2C and PPO are the clipped surrogate objective and multiple optimisation epochs per rollout — approximately 15 lines of code and no new data structures.

3. **PPO is strictly more robust.** A2C applies unconstrained policy updates, which can cause large destructive steps — exactly the failure mode observed with REINFORCE. PPO's clip ratio (ε = 0.2) explicitly bounds the policy change per update, an important safeguard on ComplexEnv's long-horizon sparse-reward setting.

In hindsight, even PPO struggled on this task, suggesting that the primary bottleneck was training budget and sample efficiency (where DQN's replay buffer holds a clear structural advantage), not the choice between A2C and PPO.

