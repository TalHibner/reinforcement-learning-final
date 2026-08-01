# Deep RL in MiniGrid — from pixels, from scratch

Final project for **Deep Reinforcement Learning (2026 B)**. Three agents — one from each
algorithm family — implemented from scratch in PyTorch and trained on two image-based
MiniGrid environments. The agents see only the rendered RGB image of the grid: no symbolic
state, no privileged getters, no external RL libraries for the algorithms themselves.

| | |
|---|---|
| **Deliverable** | [`2026B_FinalProjectFinal.ipynb`](2026B_FinalProjectFinal.ipynb) (run top-to-bottom, outputs included) |
| **Report** | [`report_026548446.pdf`](report_026548446.pdf) — the primary deliverable, self-contained |
| **Algorithms** | DQN (value-based), REINFORCE (policy-based), PPO-Clip (actor-critic) |
| **Observation** | raw rendered pixels → 84×84, grayscale on the simple env / RGB on the complex one |
| **Stack** | Python 3, PyTorch, Gymnasium, MiniGrid — trained on a Colab T4 |

---

## Headline results

**SimpleRoomEnv** — the pass/fail sanity environment, 200 held-out evaluation episodes:

| Algorithm | Mode | Success % | Avg sparse | Avg $G_0$ sparse | Avg shaped | Avg steps |
|---|---:|---:|---:|---:|---:|---:|
| DQN | greedy | **95.0** | +0.950 | +0.8726 | +9.36 | 14.1 |
| REINFORCE | greedy | 5.0 | +0.050 | +0.0486 | −0.45 | 95.2 |
| REINFORCE | sample | 51.5 | +0.515 | +0.3645 | +4.46 | 68.8 |
| PPO | greedy | 5.0 | +0.050 | +0.0486 | −0.45 | 95.2 |
| PPO | sample | 73.0 | +0.730 | +0.5135 | +6.74 | 55.6 |

**ComplexEnv** — the five-stage task (key → door → water → lava → goal), held-out layouts
(n = 200 for PPO v2, n = 50 for the other rows):

| Algorithm | Mode | Success % | Avg shaped | Mean stage | Median stage | Max stage | Lava death % | Avg ext. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DQN | greedy | 0.0 | −3.65 | 0.54 | 0.0 | 2 (Door) | 0.0 | 0.00 |
| REINFORCE | greedy | 0.0 | −4.00 | 0.00 | 0.0 | 0 (None) | 0.0 | 0.00 |
| REINFORCE | sample | 0.0 | −3.54 | 0.86 | 1.0 | 2 (Door) | 0.0 | 0.00 |
| PPO v1 | greedy | 0.0 | −2.34 | 1.96 | 2.0 | 3 (Water) | 0.0 | 0.00 |
| PPO v1 | sample | 0.0 | −2.20 | 2.28 | 2.0 | 3 (Water) | 0.0 | 0.00 |
| PPO v2 | greedy | 0.0 | −3.06 | 1.15 | 2.0 | 3 (Water) | 0.0 | 0.00 |
| **PPO v2** | **sample** | **31.0** | **+5.54** | **3.17** | **3.0** | **5 (Goal)** | 6.0 | 0.38 |

PPO with corrected reward shaping (**v2**) completes the whole chain — key, door, water
ferry, lava extinguished, goal — in 31% of unseen layouts from pixels alone. Every other
agent reaches 0%. `Avg ext.` is the mean number of lava tiles extinguished per episode; at
these counts single-episode events are anecdotes, not rates, and the report treats them
that way.

## Three findings the numbers alone don't show

1. **A two-constant change to the reward shaping decided the project.** PPO went from 0
   successes in 4,000 episodes to 357 in 5,000. Global reward-budget safety (total shaping
   mass < discounted goal payoff) and local risk–reward balance turn out to be independent
   constraints — satisfying only the first is not enough. The notebook ships the audit
   (`audit_shaping_mass`) that checks the first and the ablation that found the second.
2. **Comparable scores, categorically different mechanisms.** Measured against a fixed
   policy sampling from each agent's own action frequencies, 88% of DQN's SimpleRoomEnv
   performance comes from conditioning on the observation, while 93% of PPO's comes from a
   state-independent action prior. Two agents "solving" an environment can be doing
   entirely different things.
3. **Greedy evaluation understates both families.** For DQN, 5% action noise lifts mean
   stage from 0.54 to 1.60; for PPO v2, sampling instead of argmax lifts it from 1.15 to
   3.17 and success from 0% to 31%. The notebook reports greedy *and* sampled modes
   everywhere rather than picking the flattering one.

---

## The environments

Both are fixed by the assignment — only `max_steps` may be overridden, everything else is
done with wrappers.

| Environment | Layout | Win condition | Horizon |
|---|---|---|---|
| `SimpleRoomEnv` | empty 10×10 room | step onto the green goal | 100 steps |
| `ComplexEnv` | two rooms, locked door, water balls, lava ring | pick up the key → toggle the door → drop the key → ferry water balls → toggle each onto lava → reach the goal | 400 steps |

Observations are raw full-grid RGB (`uint8`, `H·tile × W·tile × 3`); the action space is
MiniGrid's standard `Discrete(7)`, narrowed by wrapper to 3 actions on the simple env and
6 on the complex one. Stepping on un-extinguished lava is lethal.

## What's implemented

**Agents** (from scratch, PyTorch only for the networks)
- **DQN** — target network, replay buffer, ε-greedy schedule, truncation handled correctly
  in the Bellman target (a time limit is not a terminal state).
- **REINFORCE** — Monte-Carlo returns with a running-mean baseline, entropy regularisation
  on the same scale as the policy term, and a documented early-stop rule. Includes a
  controlled ablation of the baseline choice.
- **PPO-Clip** — GAE(λ), shuffled minibatches, LR annealing, KL early-stop checked *before*
  the optimiser step, and bootstrapping through time-limit truncation rather than treating
  it as episode end.

All three share a Nature-DQN style CNN (3 conv layers → 512-unit FC head).

**Infrastructure**
- Composable `gym` wrappers for preprocessing and reward shaping, with a
  `SparseReturnTracker` as the innermost wrapper so the true sparse return is reported
  alongside the shaped one — everywhere, without cosmetics.
- One model-selection rule applied identically to all three algorithms (`BestCheckpointer`:
  rolling success rate, tie-broken by mean stage then mean shaped return).
- Periodic greedy (ε=0) probes, so training curves and evaluation measure the same policy.
- Per-episode deterministic env seeding with disjoint train / eval / video / probe seed
  spaces.
- Mid-training and converged rollout videos embedded in the notebook.
- Colab autosave + recovery (below).

---

## Running it

### Colab (as submitted)

Open the notebook, set the runtime to a **T4 GPU**, and Run All. The first cell installs
anything missing and mounts Drive if available; `STORAGE_ROOT` resolves to mounted Drive →
Colab session storage → a local folder, so the notebook also runs off-Colab unchanged.

Artifacts are written under `STORAGE_ROOT/artifacts/`:

```
artifacts/
├── logs/         <run>.json           per-episode training curves
├── checkpoints/  <run>_best.pt        best-so-far weights (what results are evaluated on)
│                 <run>_latest.pt      current weights + step counter (for resuming)
│                 <run>_final.pt       end-of-run weights
└── plots/        <run>_*.png          every saved figure
```

### Skip training entirely

Full training is several GPU-hours. The **Reload from Checkpoints** cell rebuilds all six
agents and loggers from saved artifacts:

```python
LOAD_FROM_ARTIFACTS = True     # then Run All
```

Every figure, table and video regenerates from stored state in minutes. Which weights get
loaded is stated explicitly — `prefer_best=True` uses the same best-so-far rule as the
reported results.

### Surviving a disconnect

Free-tier sessions drop mid-cell and the ComplexEnv runs are multi-hour, so every training
loop autosaves its curve and its current weights (plus `agent.steps`, which the ε schedule
and target sync depend on) every 250 episodes. A disconnect costs at most 250 episodes of
curve data. The **Recovery** section reports what is recoverable and can resume DQN
training from where it stopped, continuing the episode-seed sequence and splicing the logs
so the cumulative-steps axis stays exact:

```python
RESUME_RUN = "dqn_complex"     # None = just print the recovery status table
```

On-policy runs (REINFORCE, PPO) restart rather than splice — the caveats are documented in
the notebook and the report.

### Requirements

`gymnasium`, `minigrid`, `torch`, `numpy`, `matplotlib`, `opencv-python`, `imageio` +
`imageio[ffmpeg]`, `ipywidgets`, and `pyvirtualdisplay` on headless Linux. The install cell
is idempotent — it only pip-installs what isn't already importable.

## Reproducibility

Master seed **42** throughout. `set_seed` covers Python / NumPy / torch (+ cuDNN
deterministic) at the top of every run; each episode resets with
`episode_seed(42, episode_idx, phase)` where the phases occupy disjoint ranges
(train +0, eval +10M, video +20M, probe +30M, eval-torch +40M) so evaluation layouts are
never training layouts. Sampled-mode evaluation reseeds the torch RNG per episode, so even
the stochastic results reproduce. Every hyperparameter is stated explicitly in the
dedicated **Best Settings** cell — nothing is left to a default.

---

## Repository contents

```
2026B_FinalProjectFinal.ipynb   the submitted notebook (all outputs included)
report_026548446.pdf            the report — primary deliverable
report_images/                  figures used in the report
Assignment3.md / .pdf           the assignment specification
env_explorer.py                 button-driven inline MiniGrid explorer (drive an env by hand)
drafts/                         earlier experiment notebooks kept for provenance
```

`env_explorer.py` is a standalone debugging aid: it renders any MiniGrid env inline with
buttons, auto-discovers the env's state getters, and shows shaped reward per action — used
while designing the wrappers, not part of the training pipeline.

## Author

**Tal Hibner** — ID 026548446.

Coursework submitted for Deep Reinforcement Learning (2026 B). Published for reference;
if you are taking this course, read your institution's academic-integrity policy before
looking further.
