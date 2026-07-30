## Network Architecture Details

All three algorithms share the same **Nature-DQN-style CNN backbone** (`CNNFeatureExtractor`)
for feature extraction from pixel observations, with algorithm-specific heads on top.

### Shared CNN Backbone (`CNNFeatureExtractor`)

| Layer | Type | Filters / Units | Kernel | Stride | Activation |
|-------|------|-----------------|--------|--------|------------|
| 1 | Conv2d | 32 | 8x8 | 4 | ReLU |
| 2 | Conv2d | 64 | 4x4 | 2 | ReLU |
| 3 | Conv2d | 64 | 3x3 | 1 | ReLU |
| 4 | Flatten | -- | -- | -- | -- |
| 5 | Linear | 512 | -- | -- | ReLU |

- **Input shape (SimpleRoomEnv):** `(1, 84, 84)` -- grayscale, 1 channel
- **Input shape (ComplexEnv):** `(3, 84, 84)` -- colour, 3 channels

The CNN output is a 512-dimensional feature vector shared across all heads.

### DQN: `DQNNetwork`
- **Backbone:** `CNNFeatureExtractor` (512-d features), PyTorch default initialisation
- **Q-head:** `Linear(512, n_actions)` -- outputs Q-values for each action
- **No separate target head;** a full copy of the network serves as the target network, synced every K environment steps.

### REINFORCE: `PolicyNetwork` (+ optional `ValueNetwork`)
- **Backbone:** `CNNFeatureExtractor` (512-d features)
- **Policy head:** `Linear(512, n_actions)` producing logits for `Categorical(logits=...)`
- **Baseline:** running return statistics by default (no value head). With
  `baseline="value"` a separate `ValueNetwork` (own backbone + scalar head) is
  trained by MSE on the Monte-Carlo returns -- this is the ablation reported below.

### PPO: `ActorCriticNetwork`
- **Backbone:** `CNNFeatureExtractor` (512-d features) -- **shared** between actor and critic
- **Actor head:** `Linear(512, n_actions)` -- logits fed to `Categorical` distribution
- **Critic head:** `Linear(512, 1)` -- scalar state-value estimate
- **Initialisation:** orthogonal, trunk gain `sqrt(2)`, actor gain `0.01`, critic gain `1.0`.
  The small actor gain makes the initial logits near-zero, so the initial policy is
  near-uniform and initial entropy is `~ln|A|` by construction.
- **Design choice:** sharing the CNN backbone between actor and critic is more parameter-efficient and provides useful gradient signal from both objectives to the shared feature layers. This is standard practice in PPO (Schulman et al., 2017).

### Why This Architecture?
- The **Nature-DQN CNN** (Mnih et al., 2015) is proven for learning from pixel observations in grid-like environments.
- **84x84 input** is the standard resolution that balances detail with computational cost.
- **512-d feature layer** provides enough capacity for the relatively simple MiniGrid state space without overfitting.
- **Shared backbone for PPO** lets the critic's value loss regularise the feature extractor, improving sample efficiency.

### Preprocessing: Do We Stack Frames? **No — and here is why**

Frame stacking exists to restore the Markov property when a single frame hides
state (velocity in Atari, occluded objects, partial observability). Neither
environment here needs it, because **every latent task variable is a
deterministic function of the current frame**:

| Latent variable | How a single frame determines it |
|---|---|
| Agent position / orientation | The agent triangle is rendered with its facing |
| Door open / closed / locked | The door tile is rendered differently in each state |
| Lava extinguished so far | `n_lava - (visible lava tiles)`, with `n_lava = 3` always |
| Goal / key / water positions | Drawn directly whenever the object is on the grid |
| Carrying the key | A key is *only ever* on the floor or in the agent's hand, so "no key visible" ⟺ carried |
| Carrying a water ball | Counting: `placed = visible + consumed + carried`, where `consumed = n_lava - visible lava` and `placed` is fixed by the layout (itself visible) — so `carried ∈ {0, 1}` is determined |

Both envs use `render(highlight=False)` of the **whole grid** (not the agent's
7x7 view), so nothing is occluded: the pixel process is Markov, and stacking `k`
frames would multiply the input channels — and the first conv layer's parameters
and the replay buffer's memory — by `k` for zero information gain.

Two honest caveats, stated because they are the interesting part:
1. *Decodable in principle* is not *decoded in practice*. The 10x10 grid is
   rendered at 32 px per tile (320x320) and then resized to 84x84, so one tile is
   ~8x8 pixels and the network has to learn the counting relation above to read
   "am I carrying water". If it fails, the failure is one of **resolution and
   capacity**, not of temporal context — stacking more copies of an unresolvable
   pixel adds nothing. The right lever there is resolution or a cropped/zoomed
   observation, which is why stacking was declined rather than tuned.
2. Grayscale is applied **only on SimpleRoomEnv** (where the sole object is the
   goal). ComplexEnv keeps RGB, because there colour *is* the object identity:
   key, ball, door, lava and goal are distinguished by hue, and collapsing them
   to luminance would genuinely destroy state.

### Memory: `uint8` Replay Buffer, One Copy per Observation
Storing 100,000 transitions of 84x84x3 images as `float32` would need ~16.9 GB.
Two changes bring that down to ~2.12 GB:
1. **`uint8` storage** (1 byte per pixel instead of 4) -- a 75% saving, with the
   round-trip `round(255*x)/255` exact because preprocessing resizes on uint8
   pixels *before* normalising.
2. **No duplicated next-states.** `next_state[t]` *is* `state[t+1]` for every
   transition that does not end an episode, so only boundary next-states (one per
   episode) are stored explicitly -- halving what a naive
   `(state, next_state)` layout costs. The arrays are preallocated, so memory is
   flat rather than growing with a Python list of tuples.

### Numerical Stability: Logits vs Softmax Probabilities
During the implementation of our policy-gradient agents, we corrected a numerical stability flaw in how action distributions were sampled. Originally, the `PolicyNetwork` returned raw probabilities using `F.softmax`, which were then fed into `torch.distributions.Categorical(probs=...)`. While mathematically correct, passing very small probability values into `Categorical` can lead to underflow and unstable `log_prob` computations. We refactored the policy networks to output raw unnormalised **logits** and pass them to `Categorical(logits=...)`. This handles the log-sum-exp trick internally, providing significantly better numerical stability, particularly when policies become highly confident. Every diagnostic in the notebook (including the policy-confidence probe) uses the `logits=` path.
