# ---------------------------------------------------------------------------
# Replay buffer (DQN) — circular, preallocated, uint8 observations, one copy
# of each observation (next_state is looked up, not duplicated).
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Fixed-capacity circular buffer for (s, a, r, s', done) transitions.

    Memory layout: ``next_state[t]`` *is* ``state[t+1]`` for every transition that
    does not end an episode, so only the boundary next-states (one per episode:
    goal, lava death, or time limit) are stored explicitly. At capacity=100k with
    3x84x84 observations that is ~2.12 GB instead of the ~4.23 GB a two-copy
    layout needs (and ~16.9 GB if observations were kept as float32). The arrays
    are preallocated, so memory is flat from the first push instead of growing.

    uint8 round-trip: preprocessing resizes on uint8 pixels *before* normalising,
    so every observation value is exactly k/255 and ``round(255*x)`` recovers k
    exactly. ``np.round`` (rather than truncation) is what keeps that lossless if
    an upstream change ever produces genuinely float-valued observations.
    """

    def __init__(self, capacity=10_000):
        self.capacity = int(capacity)
        self.obs_shape = None
        self.states = None                                          # allocated on first push
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)       # true terminals only
        self.boundaries = np.zeros(self.capacity, dtype=bool)        # episode ended here
        self.boundary_next = {}                                     # idx -> stored next obs
        self.pos = 0
        self.size = 0

    @staticmethod
    def _to_storage(obs):
        return np.clip(np.round(np.asarray(obs) * 255.0), 0, 255).astype(np.uint8)

    @staticmethod
    def _to_float(arr):
        return arr.astype(np.float32) / 255.0

    def push(self, state, action, reward, next_state, done, boundary=None):
        """Store one transition.

        ``done`` is the Bellman terminal flag (termination only — never a time
        limit). ``boundary`` says whether the episode ended here for *any* reason
        (termination or truncation); it defaults to ``done`` but the training loop
        passes ``terminated or truncated`` so truncated next-states are kept too.
        """
        state = np.asarray(state)
        if self.states is None:
            self.obs_shape = state.shape
            self.states = np.zeros((self.capacity, *self.obs_shape), dtype=np.uint8)
        if boundary is None:
            boundary = bool(done)

        idx = self.pos
        self.boundary_next.pop(idx, None)          # slot is being overwritten
        self.states[idx] = self._to_storage(state)
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.dones[idx] = float(done)
        self.boundaries[idx] = bool(boundary)
        if boundary:
            self.boundary_next[idx] = self._to_storage(np.asarray(next_state))

        self.pos = (idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _sample_indices(self, batch_size):
        """Uniform indices, excluding the newest transition unless its next-state
        was stored explicitly (for the newest non-boundary slot, ``state[pos]``
        has not been written yet / is stale)."""
        newest = (self.pos - 1) % self.capacity
        exclude = self.size > 1 and not self.boundaries[newest]
        if not exclude:
            return np.random.randint(0, self.size, size=batch_size)
        out = np.empty(batch_size, dtype=np.int64)
        filled = 0
        while filled < batch_size:
            cand = np.random.randint(0, self.size, size=batch_size - filled)
            cand = cand[cand != newest]
            out[filled:filled + len(cand)] = cand
            filled += len(cand)
        return out

    def sample(self, batch_size):
        idx = self._sample_indices(batch_size)
        s = self.states[idx]
        s2 = self.states[(idx + 1) % self.capacity].copy()
        for j in np.nonzero(self.boundaries[idx])[0]:
            s2[j] = self.boundary_next[int(idx[j])]
        return (self._to_float(s),
                self.actions[idx],
                self.rewards[idx],
                self._to_float(s2),
                self.dones[idx])

    def __len__(self):
        return self.size


# ---------------------------------------------------------------------------
# Rollout buffer (PPO) — stores one rollout then is cleared
# ---------------------------------------------------------------------------
class RolloutBuffer:
    """Stores a single PPO rollout; cleared after each update.

    ``terminated`` and ``truncated`` are kept separately: the first zeroes the
    bootstrap in the TD residual, the second only cuts the GAE trace while still
    bootstrapping through V(s_T) (the time limit is a data-collection artifact,
    not environment dynamics). ``boundary_values`` holds V(s_T) for the truncated
    steps, computed once per episode end by the training loop.
    """

    def __init__(self):
        self.clear()

    def add(self, state, action, log_prob, reward, value,
            terminated=0.0, truncated=0.0, boundary_value=0.0):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.terminateds.append(float(terminated))
        self.truncateds.append(float(truncated))
        self.boundary_values.append(float(boundary_value))

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.terminateds = []
        self.truncateds = []
        self.boundary_values = []

    def __len__(self):
        return len(self.states)


# ---------------------------------------------------------------------------
# Weight initialisation (used by the PPO networks)
# ---------------------------------------------------------------------------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialisation, the PPO reference default.

    Trunk layers use gain sqrt(2) (the ReLU-preserving gain); the policy head uses
    gain 0.01 so initial logits are ~0, i.e. the initial policy is near-uniform and
    initial entropy is ~ln|A| by construction rather than by luck; the value head
    uses gain 1.0.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# ---------------------------------------------------------------------------
# CNN feature extractor (Nature-DQN style)
# ---------------------------------------------------------------------------
class CNNFeatureExtractor(nn.Module):
    """3 conv layers -> flatten -> FC 512.  Works for any in_channels."""

    def __init__(self, in_channels, feature_dim=512, orthogonal=False):
        super().__init__()
        init = layer_init if orthogonal else (lambda layer, **kw: layer)
        self.conv = nn.Sequential(
            init(nn.Conv2d(in_channels, 32, 8, stride=4)), nn.ReLU(),
            init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            self._conv_out = int(self.conv(dummy).view(1, -1).size(1))
        self.fc = nn.Sequential(init(nn.Linear(self._conv_out, feature_dim)), nn.ReLU())
        self.feature_dim = feature_dim

    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))


# ---------------------------------------------------------------------------
# DQN network — CNN + linear Q-head
# ---------------------------------------------------------------------------
class DQNNetwork(nn.Module):
    def __init__(self, in_channels, n_actions, feature_dim=512):
        super().__init__()
        self.features = CNNFeatureExtractor(in_channels, feature_dim)
        self.q_head = nn.Linear(feature_dim, n_actions)

    def forward(self, x):
        return self.q_head(self.features(x))


# ---------------------------------------------------------------------------
# Policy network (REINFORCE) — CNN + logits head
# ---------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, in_channels, n_actions, feature_dim=512):
        super().__init__()
        self.features = CNNFeatureExtractor(in_channels, feature_dim)
        self.head = nn.Linear(feature_dim, n_actions)

    def forward(self, x):
        return self.head(self.features(x))   # logits


# ---------------------------------------------------------------------------
# Value network — state-value baseline V(s) for REINFORCE (see the ablation cell)
# ---------------------------------------------------------------------------
class ValueNetwork(nn.Module):
    """State-value baseline for REINFORCE: V(s).

    Any state-only baseline leaves the policy gradient unbiased, because
    E[grad log pi(a|s) b(s)] = b(s) * grad sum_a pi(a|s) = b(s) * grad 1 = 0,
    and b(s) ~ V(s) is near variance-optimal.
    """

    def __init__(self, in_channels, feature_dim=512):
        super().__init__()
        self.features = CNNFeatureExtractor(in_channels, feature_dim)
        self.value_head = nn.Linear(feature_dim, 1)

    def forward(self, x):
        return self.value_head(self.features(x)).squeeze(-1)


# ---------------------------------------------------------------------------
# Actor-Critic network (PPO) — shared CNN, separate heads, orthogonal init
# ---------------------------------------------------------------------------
class ActorCriticNetwork(nn.Module):
    def __init__(self, in_channels, n_actions, feature_dim=512):
        super().__init__()
        self.features = CNNFeatureExtractor(in_channels, feature_dim, orthogonal=True)
        self.actor = layer_init(nn.Linear(feature_dim, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(feature_dim, 1), std=1.0)

    def forward(self, x):
        feats = self.features(x)
        return self.actor(feats), self.critic(feats)

    def get_action_and_value(self, x, action=None):
        feats = self.features(x)
        logits = self.actor(feats)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(feats)


print("Buffers and networks loaded.")
