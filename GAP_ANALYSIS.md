# Gap Analysis — `2026B_FinalProjectRunPPOcomplexA.ipynb` vs `Assignment3.md`

Scope: every code cell below the "Your Code Below" divider, the setup cells, the stored
outputs/figures, and the Discussion cell, checked line-by-line against the assignment
specification. Cell numbers are the notebook's 0-based JSON cell indices. Every finding below
was verified against the actual cell source and/or the stored execution outputs — nothing here
is speculation.

**Verdict up front:** the infrastructure is genuinely decent — honest sparse-vs-shaped
bookkeeping, correct DQN truncation handling, seeded fresh-seed greedy evaluation, stage
tracking that matches what the spec asks for. But the notebook ships with (1) a required
figure that is silently wrong by a factor of ~17, (2) zero mid-training videos actually
displayed, in a cell whose stored output provably predates its current source, (3) a PPO
implementation with a textbook truncation-bootstrapping bug that the DQN loop in the same file
explicitly avoids, (4) two of three algorithms failing the assignment's pass/fail sanity
environment with the failure explained but never fixed, and (5) a Discussion that documents a
notebook section and an ablation that do not exist. Several of these are exactly the things a
grader checks first.

---

## Addendum — lecturer clarifications (received 2026-07-29)

Three clarifications from the lecturer, received after this analysis was written, change the
severity of several findings. The original text is preserved below; affected findings carry an
**[amended]** marker pointing back here.

1. **SimpleRoomEnv does not need to be solved.** "It's mostly a sanity check for you … I
   mostly want there to be graphs to show that there is a learning process, but it doesn't
   have to reach convergence." → **C1 is downgraded from CRITICAL to HIGH.** The compliance
   failure disappears (learning-process graphs exist for all three algorithms). The
   *implementation* finding stands: the lecturer still expects DQN/PPO to solve it "relatively
   quickly", and PPO's 5% greedy vs 56% sampled gap — with the REINFORCE/PPO greedy rows
   bit-identical — remains evidence of the M1/M3 defects, not an acceptable end state.
2. **Two graphs are explicitly requested: discounted reward vs episode, and num steps vs
   episode.** Steps-per-episode already exists in every training figure → that half is
   already satisfied. Discounted return per episode exists nowhere and the logger does not
   record it → **new finding C8 below.** The lecturer's stated purpose (detect
   "reaching the goal faster" after the success rate plateaus) also motivates a
   success-conditioned episode-length curve — see the blueprint revision.
3. **Submission format: organized code files are fine, plus a Colab notebook with all
   relevant outputs that may simply load saved logs and display them; the report is the
   primary deliverable.** → The falsely-claimed "Reload from Checkpoints" section (C6) is no
   longer merely a claim to delete — it is now the *requested workflow* and must be
   implemented. This also legitimizes load-and-display over re-training in the submitted
   notebook, easing the Run-All burden noted in Phase 0 of the blueprint.

---

## 1. CRITICAL — spec deliverables missing, wrong, or fabricated

### C1. Two of three algorithms fail the pass/fail sanity check, and the failure is analyzed instead of fixed
**[amended — downgraded to HIGH per lecturer clarification #1; the compliance framing below
no longer applies, the implementation diagnosis still does]**
- Spec §1.1: SimpleRoomEnv is "a pass/fail sanity check … If an agent cannot solve this, the
  problem is in your preprocessing, network, hyperparameters, or training loop."
- Cell 51 (your own markdown): "all three algorithms should converge to >90% success rate."
- Actual results (cells 55, 57, 61): REINFORCE final training SR100 = 0.49, greedy eval
  **5.0%**; PPO training SR100 plateaued at ~0.63 from episode ~250 onward (2,250 further
  episodes, zero improvement — see the flat success-rate panel in the PPO training figure) and
  greedy eval **5.0%**. Only DQN passes (90.5% greedy).
- The Discussion reframes this as a greedy-vs-sampled insight. The diagnosis is fine; the spec
  is explicit that this outcome means the pipeline is broken, and the notebook proceeds to
  ComplexEnv with the broken pipeline anyway. Sampled evaluation at 46–56% with a 100-step cap
  is not "solved" by any reading.
- Damning detail: the REINFORCE-greedy and PPO-greedy eval rows are **identical at printed
  precision across all four statistics** (5.0%, +0.050, −0.45, 95.2 steps over the same 200
  seeds — cell 61 output). Two independently trained networks matching mean episode length to
  0.1 steps over 200 episodes is not plausible unless both argmax policies collapsed to the
  same degenerate near-constant behavior (e.g. "always forward": ~5% of random spawns face the
  goal line; failures truncate at 100 steps → mean ≈ 95). The converged videos corroborate:
  both policy-gradient SimpleRoomEnv videos end at R = −1.00 (cell 80 output). This is a
  policy-collapse bug being reported as a finding.

### C2. The required cumulative-environment-steps curve is wrong for DQN/ComplexEnv — by ~17×
- Spec §3 requires "Cumulative environment steps versus episode … Include it for both
  environments" and explains it exists to "place every run on a common budget axis."
- Cell 67 defines `_merge_logger`, which merges `("rewards", "sparse_rewards", "steps",
  "successes", "stages")` across the 200-episode training chunks — and silently drops
  `cumulative_steps` and `total_steps`.
- Consequence, visible in the stored figures: in "DQN — ComplexEnv" (cell 67 output) the
  cumulative-steps panel spans **200 episodes / 80,000 steps** while the other three panels of
  the same figure span **4,000 episodes (~1.35M steps)**. In the ComplexEnv comparison figure
  (cell 73 output), DQN's budget line is an invisible stub near the origin while PPO's line
  reaches 1.6M — a reader comparing sample budgets concludes DQN used ~5% of PPO's steps when
  it actually used ~85% of them. The exact chart the spec added to prevent budget distortion
  is the one that distorts it.
- The corruption is persisted: `save_logger(dqn_complex_log, "dqn_complex")` writes
  `cumulative_steps` (200 entries) and `total_steps = 80000` to the artifacts JSON.

### C3. No mid-training videos are actually shown, and cell 80 was edited after its last execution
- Spec rule 5: "Include short video clips of the agent partway through training and after it
  has converged (in the notebook)."
- Cell 80's stored output lists **all six** mid-training entries as
  "not found (training may not have reached this episode)". Zero partway clips are displayed
  in the submitted notebook.
- Worse: the stored output's labels ("ep 250/500", "ep 1000/2000") do not exist anywhere in
  the current cell source, whose labels are "ep 750/2000", "ep 1500/3000", etc. Stream output
  can only come from the source that executed — so the source was rewritten after the run and
  never re-executed. The training cells (53, 55, 57, 67, 71) did record
  `*_mid750/mid1500/mid1250/mid2000.mp4` to Drive; the old display cell looked for the old
  filenames, found nothing, and nobody noticed. Cell 79's markdown ("episode 250 for
  SimpleRoomEnv, episode 1000 for ComplexEnv") is stale text from the same earlier revision
  and contradicts both the code and the video labels one cell down. Rule 8 territory: a cell
  whose source and output visibly disagree is what "edited notebook" flags look like.
- Independently: REINFORCE/ComplexEnv has `video_at=None` (cell 69), so its partway clip was
  never even recorded. 5 of 6 recorded, 0 of 6 displayed.
- The converged videos that *are* embedded show: REINFORCE-simple −1.00, PPO-simple −1.00,
  and all three ComplexEnv agents at −4.00 — i.e. 400 steps of pure step-penalty, not even a
  key pickup, from the "best" checkpoints on the fixed video seed.

### C4. No dedicated "best settings" cell
- Spec §5: "In a dedicated notebook cell, clearly state the best training and inference
  settings you found." No such cell exists. Hyperparameters are scattered across five training
  cells and prose in cell 81. A grader scanning for the required cell finds nothing.

### C5. Frame-stacking is never addressed
- Spec §2.2 explicitly asks for "whether you stack frames and why." The preprocessing is
  described (84×84, grayscale vs color, CHW, [0,1]) in cells 33/47/81, but frame stacking is
  mentioned nowhere — not adopted, not declined, not discussed. On a task where the Discussion
  itself says the agent "must infer latent task state … from pixels", silently ignoring the
  one preprocessing question the spec names is a visible gap.

### C6. The Discussion documents deliverables that do not exist in the notebook
- Cell 81, "Key Design Choices": "we provided an optional **'Reload from Checkpoints'**
  section that allows anyone reviewing the notebook to instantiate the agents directly from
  these saved artifacts." **There is no such section.** The only "reload" in the file is the
  inline best-checkpoint reload inside cell 67.
- Cell 81: "A learned state-value baseline was tested for REINFORCE … it degraded performance
  … we discuss the value-baseline version as an unsuccessful ablation." **No code, no run, no
  number for this ablation exists in the notebook.** Its only trace is `ValueNetwork`
  (cell 40) — defined, never instantiated — and a `value_net` key in `save_agent` for an
  attribute no agent has. An ablation claim with zero evidence in the deliverable is worse
  than no ablation: it invites the question of what else in the Discussion isn't backed by
  the notebook.
- Cell 47 contradicts cell 40 the same way: "No value head; the mean discounted return over
  the episode serves as the baseline" — while a class named "State-value baseline for
  REINFORCE: V(s)" sits unused forty lines up.
- **[amended per lecturer clarification #3]** The "Reload from Checkpoints" section is now the
  *requested submission workflow* (display-only notebook loading saved logs/checkpoints), so
  the resolution is: implement it, not delete the claim.

### C7. The notebook is not clean (spec rule 6), and some of the mess is self-incriminating
- Cell 69 ships with pasted advisory comments in the code:
  "`#Given REINFORCE ComplexEnv has never shown progress, you can safely cap it at 500 or even
  300 episodes for a symbolic run`" and "`#This is defensible in the report because REINFORCE
  ComplexEnv has repeatedly stalled…`". That is somebody's advice about how to *present* the
  run, left inside the submitted artifact, admitting the 300-episode run is symbolic (vs 4,000
  episodes for DQN/PPO — a 13× budget gap in a graded three-way comparison).
- Cell 67 contains instructions to a *previous author of the cell*: "NOTE: This assumes
  train_dqn accepts an existing_agent argument … If your train_dqn does not accept this, use
  the alternative approach in the next section." There is no "next section" / "alternative
  section below." Two such notes remain.
- The Algorithm Comparison table in cell 81 has a malformed row — the ComplexEnv line carries
  two orphaned cells `| None (0 / 50) | None (0 / 50) |` left over from an earlier draft of
  the table (the draft text survives verbatim in `ppo_dump.md`).
- Dead code below the divider: `ValueNetwork` (cell 40), `evaluate_agent` (cell 49 — fully
  superseded by `evaluate_agent_mode`, never called), the `len(rec) == 3` video-return branch
  in `train_dqn`/`train_ppo` (impossible: `record_agent_video` always returns a 2-tuple),
  `import copy` (cell 67, unused), `from types import SimpleNamespace` (cell 8, unused),
  `import json` three times (cells 31, 61, 76), `import numpy` again mid-notebook (cell 50).

### C8. [added 2026-07-29] No discounted-return-per-episode graph, and the logger cannot produce one
- Lecturer clarification #2 explicitly requests "discounted reward VS episode" alongside
  steps-per-episode (the latter already exists in every training figure).
- No discounted return is plotted anywhere, and `TrainingLogger` (cell 31) records only
  *undiscounted* per-episode sums (`rewards`, `sparse_rewards`) — the quantity cannot be
  derived from the saved logs in general.
- Partial rescue for the runs already on disk: the **sparse** reward is nonzero only at the
  terminal goal step, so the sparse discounted return is exactly
  G₀ = success · γ^(steps−1), and `steps`/`successes` are logged. The lecturer's plot can
  therefore be reconstructed for every existing run *for the sparse signal* without
  retraining. The **shaped** discounted return is not reconstructible (per-step rewards were
  never stored) — future runs must accumulate G₀ ← G₀ + γ^t·r online in the logger.
- The lecturer's stated purpose (is the agent "reaching the goal faster" after the success
  rate plateaus?) is served even more directly by a success-conditioned episode-length curve
  (rolling mean of `steps` over successful episodes only) — same logged data, one extra line
  of plotting. Both belong in the standard figure set.

---

## 2. HIGH — mathematical and algorithmic defects

### M1. PPO treats truncation as termination; the DQN loop ten lines up handles it correctly
- Cell 49, `train_dqn`: "`# Do not treat truncation as a true terminal for the Bellman
  target`" → `done_for_dqn = float(terminated)`. Correct.
- Same cell, `train_ppo`: `done_for_gae = float(terminated or truncated)`. Wrong — the bug the
  DQN comment warns about, shipped in the same file.
- Effect: at every time-limit truncation, GAE zeroes the bootstrap (`(1 − done) = 0`), telling
  the critic the world ends with value 0 at `max_steps`. On SimpleRoomEnv early training,
  most episodes truncate; on ComplexEnv **every** episode of note truncates (steps = 400
  throughout cell 71's log), so 100% of PPO-complex episode boundaries inject a false
  terminal. The critic is thereby trained to fit a *finite-horizon* value with no time input:
  the same pixel state observed at t=10 and t=390 carries wildly different targets → an
  irreducible-noise floor in the value regression → systematically noisy advantages. This is
  the standard "Time Limits in RL" (Pardo et al. 2018) failure; the fix is bootstrapping
  `γ·V(s_T)` on truncation. It is a plausible contributor to PPO's 0.63 plateau on the sanity
  env and its milestones-then-idle behavior on ComplexEnv.
- REINFORCE has the same finite-horizon bias (MC returns cut at truncation), which is inherent
  to episodic REINFORCE without a bootstrap — acceptable, but never acknowledged.

### M2. REINFORCE's loss mixes a per-episode SUM with a per-step MEAN, making the entropy bonus decorative
- Cell 44: `policy_loss = -(log_probs * returns).sum()` but
  `entropy_loss = -entropies.mean()`, combined as `policy_loss + entropy_coef * entropy_loss`.
- The policy term scales with episode length T (~100–400 terms of O(1) after return
  standardization); the entropy term is O(1) × coef 0.001–0.02. Relative weight of the
  entropy gradient: roughly `coef / T` ≈ 10⁻⁵–10⁻⁴ of the policy gradient. The "entropy
  regularisation for exploration" advertised in cell 43 is numerically a no-op. Either sum
  both (entropy × T) or mean both.
- Related: standardizing returns *within a single episode* means an all-failure episode
  (every reward = −0.01) still produces a full-magnitude gradient — after `(G − mean)/std`,
  late-episode actions get positive normalized returns and early ones negative, purely from
  discounting geometry. Zero-information episodes are converted into confident, arbitrary
  policy updates. With REINFORCE-simple failing ~80% of episodes for the first 1,800 episodes
  (cell 55 log), most of its gradient budget was structured noise.

### M3. PPO deviates from standard practice in ways that compound
- **No minibatching**: `update()` does `n_epochs` full-batch gradient steps over the whole
  rollout (512 obs on simple, 1024 on complex) — 6–8 gradient steps per 512–1024 env steps,
  no shuffled minibatches (cell 46). Standard PPO (and the reason it works with n_epochs > 1)
  uses shuffled minibatches; full-batch × few-steps is both less sample-efficient and more
  prone to the exact ratio saturation the clip is meant to manage.
- **KL early-stop fires after the step**: `approx_kl` is computed from `new_lp` evaluated
  *before* this epoch's `optimizer.step()`, and the `break` happens after the step is already
  applied — one full update is always applied beyond the threshold. SB3 checks before
  stepping.
- **No orthogonal initialization, no LR annealing, no value-loss clipping** — defaults are
  defensible individually, but combined with M1 and no minibatching, this PPO is far from
  "PPO best practices" while the markdown (cell 45) sells it as canonical PPO-Clip.
- The result is visible: on the *sanity* env PPO needs ~250 episodes to hit 0.6 and then never
  improves for 2,250 more; entropy collapses to 0.428 with the argmax wrong often enough that
  greedy eval scores 5%.

### M4. DQN target-network sync is latently coupled to `train_freq`
- Cell 42: the sync check `if self.steps % self.target_update == 0` lives *inside* `update()`,
  which is only called when `steps % train_freq == 0` (cell 49). Syncs therefore happen only
  at steps divisible by both. The shipped runs survive by arithmetic luck (500 % 4 == 0,
  1000 % 4 == 0). Your own sanity check (cell 50: `target_update=10, train_freq=4`) silently
  syncs every 20 steps — half the configured rate — and prints success anyway. Move the sync
  to the env-step loop or check `steps // train_freq` counters.

### M5. Exploration schedules undermine the headline numbers
- ComplexEnv DQN: `eps_end=0.15` forever (cell 67). The celebrated "peak SR100 = 0.33" is
  measured under a 15%-random behavior policy; greedy evaluation of the same checkpoint scores
  0%. The Discussion calls this gap "an important scientific result" — part of it is just
  measuring a different policy than the one evaluated. A periodic greedy probe (ε=0 rollouts
  every N episodes) would have separated learning progress from exploration luck, and its
  absence is why the best-checkpoint selection (see M7) chased an ε-inflated metric.
- SimpleRoom DQN: `eps_decay_steps=120_000` but the run only lasted ~112K steps — the schedule
  never reached its floor (final ε = 0.116, printed in cell 53). The documented ε_end=0.05 was
  never actually in effect.

### M6. The state-space estimate contains a concrete arithmetic error
- Cell 81: "2⁴ lava extinguishing combinations". The lava ring is 3 tiles everywhere in the
  env (`lava_ring` is a 3-tuple, `info["n_lava"] = 3`, your own shaping default
  `max_water_rewards = 3`) → 2³ = 8 subsets, not 16. The ~12.6M product also omits water-ball
  presence combinations (2³) while double-counting via the carry state, so the number is both
  internally inconsistent and mislabeled "conservative." Fine as an order-of-magnitude
  estimate; not fine to show arithmetic that contradicts the environment one screen up.

### M7. Overclaims and asymmetric methodology in the results
- "Avg Ext. of 0.02 confirms that it occasionally learned water-to-lava extinguishing"
  (cell 81): 0.02 × 50 = **one tile, once, in one episode of fifty**. n=1 "confirms" nothing;
  it is consistent with a single lucky trajectory.
- Model selection is asymmetric: DQN/ComplexEnv is evaluated at its best-SR100 checkpoint
  (ep 3600), while PPO and REINFORCE are evaluated at final weights. The comparison table
  presents these as like-for-like. (Best-checkpoint selection itself is fine — it's disclosed
  — but apply it to all three or to none.)
- The eval tables' "MaxStage" column is the *max over 50 episodes* — the single best episode —
  presented alongside means. A stage *distribution* mean/median would be the honest headline;
  the distribution exists (cell 77) but the summary table leads with the best case.

### M8. Reward-shaping mass is never audited against the goal reward
- Available milestone mass per episode: key 1 + door 2 + right-room 1 + first-water 0.5 +
  lava 3×3 = **+13.5**, vs goal = +10 (delivered ~150+ steps later under γ=0.99, i.e. worth
  ~×0.2 of its face value at decision time near the start). The shaped optimum plausibly does
  not include reaching the goal at all. Your own PPO found exactly this optimum:
  key→door→water (+0.5 net of step penalty), then idle to truncation, 400 steps, forever
  (cell 71: steps=400 on every logged line, shaped ≈ +0.1…+0.5, sparse 0.00). The Discussion
  analyzes water-farming carefully but never runs this total-mass sanity check, which the
  spec's "think about what your shaping actually encourages" is pointing at.
- Letter-of-the-law issue: `RichShapingWrapper` calls `core.lava_positions()` — a *positions*
  getter — inside reward logic (the `drop_water_penalty` guard and gated modes, cell 33). The
  env docstring says positions getters are "for analysis, not for shaping." The quantity you
  need is legally available as `info["n_lava"] − extinguished_lava_count()`. No geometry
  leaks (it's just a count), but it's the one line a strict grader can cite as touching a
  forbidden getter class in shaping.

---

## 3. MEDIUM — memory and engineering

### E1. The replay buffer stores every observation twice — 4.2 GB at the configured capacity
- Cell 40: each transition stores `state` **and** `next_state` as independent uint8 arrays.
  At `buffer_size=100_000` (cell 67) with 3×84×84 obs: 2 × 21,168 B × 100K ≈ **4.23 GB** —
  the single largest allocation in the project, double what a standard
  next-obs-linked layout needs (2.12 GB). `next_state[t]` **is** `state[t+1]` for every
  non-boundary step. Cell 47 brags about the uint8 "crucial memory optimization" (75% saved)
  while leaving 50% of the remainder duplicated. Not a leak (the circular overwrite is
  correct), but calling it optimized invites this audit. The pre-fix float32 version would
  have been 16.9 GB — an OOM, which is presumably how the uint8 fix was discovered.
- Minor: `self.buffer = []` grows to capacity as a Python list of tuples (small per-item
  overhead, no preallocation), and `random.sample` on a 100K list is fine but `pos` tracking
  plus a preallocated ndarray ring would be both faster and flat.

### E2. REINFORCE holds one autograd graph per environment step for the whole episode
- Cell 44: `_log_probs` retains the full CNN forward graph for every step; on ComplexEnv
  that's 400 concurrent graphs (~tens of MB of activations per episode, GPU-resident) before
  `update()` releases them. Works at this scale; the standard store-states/recompute-in-update
  pattern is flat-memory and also fixes M2's structure. Worth a sentence in the notebook;
  there is none.

### E3. PPO updates backprop through the entire rollout at once
- 1024 × 3×84×84 float32 = **86.7 MB** for the states tensor alone, plus conv activations for
  a 1024-batch forward/backward, × 6 epochs (cells 46/71). It fits on a Colab GPU, which is
  the only reason the absence of minibatching (M3) presents as a quality issue rather than a
  crash.

### E4. Verified non-issues (credit where due)
- The uint8 round-trip (`uint8/255 → float32 → ×255 → astype(uint8)`) was tested exhaustively:
  0 of 256 values corrupted — exact, because resizing happens on uint8 *before* normalization.
  Note it is exact only by that ordering; `astype` truncates, so any upstream change to
  float-valued observations silently corrupts the buffer. `np.round` would make it robust.
- GAE episode-boundary masking inside rollouts is correct; the final-state bootstrap after the
  rollout loop is correctly masked by the last `done`; PPO's old-log-probs are detached floats
  (no graph leak); DQN's warmup/sync interaction can't misfire at the shipped configs.

### E5. The storage-root logic fights itself and breaks off-Colab reruns
- Cell 5 computes `STORAGE_ROOT` with a Drive-or-local fallback. Cell 10 then hardcodes
  `STORAGE_ROOT = "/content/drive/MyDrive/rl_final_project"`, clobbering the fallback. On any
  machine without that path (any grader's local rerun — rule 8 says reruns happen),
  `video_path()` → `os.makedirs("/content/drive/...")` → crash or garbage under `/content`.
  Cell 31 rebuilds `ARTIFACTS` from the clobbered value, so the cell-5 logic is 100% dead.

### E6. Evaluation churns environments
- `evaluate_agent_mode` constructs and closes a fresh wrapped env *per episode* (200 per eval
  run; 5 runs on simple). Harmless at MiniGrid scale, wasteful as a pattern; one env per run
  with per-episode `reset(seed=…)` is the idiom (and is exactly what the training loops do).

---

## 4. LOW — hygiene

- `policy_confidence` (cell 62) builds `Categorical(probs=softmax(logits))` — the precise
  probs-antipattern cell 47 devotes a paragraph to having eliminated ("we refactored the
  policy networks to output raw unnormalised logits").
- Comment bug in cell 33: the water-pickup detection block is titled `# Key pickup`.
- `dropped_water_now` hardcodes `int(action) == 4`; correct only because the shaping wrapper
  sits inside `ActionSubsetWrapper` *and* the subset is the identity on 0–5. Reorder the
  wrappers or change the subset and it breaks silently. Compare against
  `self.env.unwrapped.actions.drop`.
- Chunked DQN training prints local episode numbers ("DQN ep 100/200" forty times, cell 67) —
  the global episode is unrecoverable from the log without counting blocks.
- `save_agent` saves a `value_net` slot no agent defines; `episode_seed`'s phase offsets are
  fine, but nothing prevents a chunk-seed base (`42 + episodes_done`) from colliding with
  another run's range — it happens not to, by inspection, which is the wrong kind of
  guarantee.
- Cell 79's "Detailed Failure-Mode Tracking" paragraph says `evaluate_agent` was updated to
  track lava deaths — the function that actually tracks them is `evaluate_agent_mode`;
  `evaluate_agent` is the dead one.

---

## 5. What actually holds up

Credit is due, and these are not small things:
- `SparseReturnTracker` as the innermost wrapper, with shaped and sparse returns reported
  separately everywhere, is exactly the right instrument against shaping self-deception — and
  the notebook uses it honestly (0.00 sparse is printed, repeatedly, without cosmetics).
- DQN's truncation handling, per-episode deterministic env seeding with disjoint
  train/eval/video seed spaces, greedy evaluation on fresh seeds, and all-hyperparameters-
  explicit training calls match the spec's reproducibility demands.
- The stage-tracking design (latched milestones → `stage_reached` in info → scatter,
  distributions, per-algorithm grouped bars, lava-death and extinguish counts) is a genuinely
  good implementation of §3's "show how far the agent gets" requirement.
- The greedy-vs-sampled dual evaluation with per-episode torch reseeding, and the
  policy-confidence diagnostic, are beyond-requirements work of real diagnostic value.
- The anti-farming `water_mode` iteration (`first`/`gated`/`per_pickup`/`per_lava`) shows real
  engagement with shaping exploits; `per_lava`'s progress-gated design is genuinely clever
  (even if the final config retreats to `first`).
- Execution counts are sequential (4→46): the notebook is an honest top-to-bottom run —
  except cell 80 (C3), which is why that one stands out.

---

## 6. Prioritized fix list

1. **Fix `train_ppo` truncation handling** (M1): store `terminated` for GAE masking and
   bootstrap `V(s_{t+1})` on truncation; then retrain PPO — this plus (2) is the likely path
   past the 0.63 plateau.
2. **Fix the SimpleRoomEnv failures before touching ComplexEnv again** (C1): the spec defines
   this env as the pipeline test and 2/3 algorithms fail it. For PPO: M1 + shuffled
   minibatches + entropy ≥ 0.01 + longer training; for REINFORCE: M2 (consistent scales, real
   entropy weight) + skip/reweight all-failure episodes.
3. **Fix `_merge_logger`** (C2): rebuild `cumulative_steps` from the merged `steps` list
   (`np.cumsum`), fix `total_steps`, regenerate the DQN-ComplexEnv figure, the comparison
   figure, and the saved JSON.
4. **Re-run cell 80 and fix cell 79's text** (C3); add `video_at` to the REINFORCE-ComplexEnv
   run so all six partway clips exist and display.
5. **Add the dedicated best-settings cell** (C4) and a frame-stacking paragraph with a reason
   (C5).
6. **Delete or implement every claimed-but-absent artifact** (C6): the "Reload from
   Checkpoints" section (implementing it is ~20 lines and genuinely useful) and the
   value-baseline ablation (either add the run + numbers or cut the paragraph and the dead
   `ValueNetwork`).
7. **Scrub the notebook** (C7): the two advisory-comment blocks in cells 67/69, the phantom
   "alternative section" notes, the malformed table row, dead code and duplicate imports,
   local-episode logging.
8. **Decouple target sync from `train_freq`** (M4); drop ComplexEnv `eps_end` to ~0.05 after
   discovery plateaus and add periodic ε=0 greedy probes so training metrics and evaluation
   measure the same policy (M5).
9. **Equalize the comparison**: same model-selection rule for all three algorithms, a
   non-symbolic REINFORCE-ComplexEnv budget (or an explicit, principled early-stop criterion
   stated in your own words), stage means not maxes in the headline table (M7).
10. **Audit shaping mass vs goal** (M8): make the discounted milestone total strictly smaller
    than the discounted goal payoff along a competent trajectory, replace `lava_positions()`
    with `n_lava − extinguished_lava_count()`, and fix the 2⁴→2³ estimate (M6).

---

## 7. Spec-compliance checklist

| Spec item | Status |
|---|---|
| §1 envs fixed, only `max_steps` overridden | PASS (cells 13/15 untouched; wrappers only) |
| §1.3 pixels-only agent input | PASS (network inputs are wrapped obs only; getters confined to wrappers/logging) |
| §1.5 no distance/geometry shaping | PASS in substance; one positions-getter call in shaping (M8) |
| §1.1 SimpleRoomEnv solved as sanity gate | **[amended]** requirement relaxed by lecturer to "graphs showing a learning process" → PASS as compliance; greedy-collapse defect remains open (C1→HIGH) |
| Lecturer #2: discounted reward vs episode | **FAIL — not plotted, not logged** (C8) |
| Lecturer #2: num steps vs episode | PASS (already in every training figure) |
| Lecturer #3: code files + display-only Colab loading saved outputs | **Not yet implemented** — now the requested workflow (C6 amendment) |
| §2.1 MDP characterization + state-space size | Present; arithmetic slip (M6) |
| §2.2 preprocessing described; frame-stack decision | **Frame stacking never addressed** (C5) |
| §2.2/§1.4 action subset stated per task | PASS (3 actions simple / 6 complex, documented) |
| §2.3 architectures in full | PASS (cell 47 table) |
| §2.4 one algorithm per family, from scratch, both envs | PASS in letter; REINFORCE-ComplexEnv is a self-described "symbolic" 300-episode run (C7/M7) |
| §2.5 exploration + hyperparameters documented | Mostly PASS; ε-schedule issues (M5) |
| §3 reward/steps/success plots per algo & env | PASS |
| §3 cumulative-steps curve, both envs | **FAIL for DQN-ComplexEnv** (C2) |
| §3 ComplexEnv stage-progress analysis | PASS (strong) |
| §4 greedy eval, fresh seeds, return/steps/SR + stages | PASS (plus sampled mode extra) |
| §5 discussion of strengths/weaknesses/shaping | PASS (with overclaims, M7) |
| §5 dedicated best-settings cell | **FAIL** (C4) |
| Rule 5 videos partway + converged | **FAIL — zero partway clips displayed; one never recorded** (C3) |
| Rule 6 clean notebook | **FAIL** (C7) |
| Rule 7 seeding + hyperparameter documentation | PASS |
| Rule 8 no post-run edits | **At risk — cell 80 output provably predates its source** (C3) |
