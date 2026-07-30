# Canonical ComplexEnv stage names, used by every stage plot/table below.
STAGE_LABELS = ["None", "Key", "Door", "Water", "Lava", "Goal"]

# ---------------------------------------------------------------------------
# Artifact persistence (ARTIFACTS / STORAGE_ROOT come from the setup cell)
# ---------------------------------------------------------------------------
# Every sub-module an agent may own. save/load walk this list and skip whatever
# a given agent does not have, so no slot is written for a module that does not
# exist (DQN has no policy, REINFORCE has a value_net only with the baseline on).
AGENT_MODULES = ("q_net", "target_net", "policy", "value_net", "network")


def _agent_state(agent):
    return {a: getattr(agent, a).state_dict()
            for a in AGENT_MODULES if getattr(agent, a, None) is not None}


def save_agent(agent, name):
    path = os.path.join(ARTIFACTS, "checkpoints", f"{name}.pt")
    torch.save(_agent_state(agent), path)
    return path


def load_agent(agent, name, map_location=None):
    """Inverse of save_agent: restore weights into an already-constructed agent."""
    path = os.path.join(ARTIFACTS, "checkpoints", f"{name}.pt")
    state = torch.load(path, map_location=map_location or device)
    for attr in AGENT_MODULES:
        module = getattr(agent, attr, None)
        if module is not None and attr in state:
            module.load_state_dict(state[attr])
    return agent


def save_figure(fig, name):
    if fig is None:                       # e.g. plot_stage_progress with no stage data
        return None
    path = os.path.join(ARTIFACTS, "plots", f"{name}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


# ---------------------------------------------------------------------------
# Seeding — call once at the top of each training run
# ---------------------------------------------------------------------------
def set_seed(seed=42):
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def episode_seed(seed, episode_idx, phase="train"):
    """Deterministic per-episode seed for env.reset().

    The phase offsets keep train / eval / greedy-probe / video seed spaces
    disjoint, so evaluation layouts are never layouts the agent trained on.
    """
    phase_offset = {"train": 0, "eval": 10_000_000, "video": 20_000_000,
                    "probe": 30_000_000, "eval_torch": 40_000_000}.get(phase, 0)
    return int(seed) + phase_offset + int(episode_idx)


# ---------------------------------------------------------------------------
# Training logger — stores per-episode metrics
# ---------------------------------------------------------------------------
class TrainingLogger:
    """Per-episode metric store for training runs.

    Per episode it records the undiscounted shaped and sparse returns, the
    *discounted* returns G_0 = sum_t gamma^t r_{t+1} for both signals, the
    episode length, the success flag, the cumulative environment-step count and
    (ComplexEnv) the furthest stage reached.

    The discounted shaped return has to be accumulated online by the training
    loop (per-step rewards are not stored). The discounted sparse return does
    not: the sparse reward fires only on the terminal goal step, so
    G_0^sparse = success * gamma^(steps-1) exactly, which is why it can also be
    reconstructed for runs logged before this field existed (see load_logger).
    """

    def __init__(self, gamma=0.99):
        self.gamma = float(gamma)
        self.rewards = []               # undiscounted shaped return
        self.sparse_rewards = []        # undiscounted true env return
        self.disc_rewards = []          # discounted shaped return (NaN if not tracked)
        self.disc_sparse_rewards = []   # discounted true env return
        self.steps = []
        self.successes = []
        self.cumulative_steps = []
        self.total_steps = 0
        self.stages = []                # ComplexEnv: max stage reached per episode
        self.probe_episodes = []        # greedy (eps=0) probe: episode index
        self.probe_successes = []       # greedy probe: success rate at that episode

    def log(self, reward, steps, success, info=None, sparse_reward=0.0,
            disc_reward=None, disc_sparse_reward=None):
        self.rewards.append(float(reward))
        self.sparse_rewards.append(float(sparse_reward))
        self.steps.append(int(steps))
        self.successes.append(float(success))
        self.total_steps += int(steps)
        self.cumulative_steps.append(self.total_steps)
        if disc_sparse_reward is None:
            disc_sparse_reward = float(success) * self.gamma ** max(0, int(steps) - 1)
        self.disc_sparse_rewards.append(float(disc_sparse_reward))
        self.disc_rewards.append(float("nan") if disc_reward is None
                                 else float(disc_reward))
        if info is not None and "stage_reached" in info:
            self.stages.append(info["stage_reached"])

    def log_greedy_probe(self, episode, success_rate):
        """Record a periodic epsilon=0 / argmax probe so the training curve and
        the reported evaluation measure the *same* policy."""
        self.probe_episodes.append(int(episode))
        self.probe_successes.append(float(success_rate))

    def get_success_rate(self, window=100):
        if not self.successes:
            return 0.0
        recent = self.successes[-window:]
        return sum(recent) / len(recent)

    def get_mean(self, attr, window=100):
        data = [v for v in getattr(self, attr, [])[-window:] if v == v]  # drop NaN
        return float(np.mean(data)) if data else 0.0


# ---------------------------------------------------------------------------

def save_logger(logger, name):
    path = os.path.join(ARTIFACTS, "logs", f"{name}.json")
    with open(path, "w") as f:
        json.dump({k: v for k, v in vars(logger).items()
                   if isinstance(v, (list, int, float))}, f)
    return path


def load_logger(name):
    path = os.path.join(ARTIFACTS, "logs", f"{name}.json")
    with open(path, "r") as f:
        data = json.load(f)

    logger = TrainingLogger()
    for k, v in data.items():
        setattr(logger, k, v)

    # Backwards compatibility with logs saved by earlier revisions of this
    # notebook: fill in / repair every derived series so a loaded log plots
    # exactly like a freshly trained one.
    n = len(logger.rewards)
    if len(logger.sparse_rewards) != n:
        logger.sparse_rewards = [0.0] * n
    if len(logger.cumulative_steps) != n:          # chunked runs used to drop this
        logger.cumulative_steps = [int(x) for x in np.cumsum(logger.steps)]
        logger.total_steps = int(logger.cumulative_steps[-1]) if n else 0
    if len(logger.disc_sparse_rewards) != n:
        g = float(getattr(logger, "gamma", 0.99) or 0.99)
        logger.disc_sparse_rewards = [
            float(s) * g ** max(0, int(st) - 1)
            for s, st in zip(logger.successes, logger.steps)
        ]
    if len(logger.disc_rewards) != n:
        logger.disc_rewards = [float("nan")] * n   # not recoverable after the fact
    return logger


def merge_logger(target, source):
    """Append a later training chunk onto `target`, keeping derived series exact.

    The previous version merged only (rewards, sparse_rewards, steps, successes,
    stages) and silently dropped cumulative_steps/total_steps, which made the
    required cumulative-environment-steps curve span the last chunk only. Both
    are now rebuilt from the merged episode lengths.
    """
    offset = len(target.rewards)
    for attr in ("rewards", "sparse_rewards", "disc_rewards",
                 "disc_sparse_rewards", "steps", "successes", "stages"):
        getattr(target, attr).extend(getattr(source, attr, []))
    for ep, sr in zip(getattr(source, "probe_episodes", []),
                      getattr(source, "probe_successes", [])):
        target.probe_episodes.append(int(ep) + offset)
        target.probe_successes.append(float(sr))
    target.cumulative_steps = [int(x) for x in np.cumsum(target.steps)]
    target.total_steps = int(target.cumulative_steps[-1]) if target.cumulative_steps else 0
    return target


# ---------------------------------------------------------------------------
# Model selection — one rule, applied identically to all three algorithms
# ---------------------------------------------------------------------------
class BestCheckpointer:
    """Best-so-far checkpointing keyed on rolling training performance.

    Every `every` episodes the score (rolling success rate, then mean stage, then
    mean shaped return over the last `window` episodes) is compared against the
    best seen; on improvement the agent's weights are written to
    ARTIFACTS/checkpoints/<name>_best.pt. `load_best` restores them before
    evaluation. Using the identical rule for DQN, REINFORCE and PPO is what makes
    the three-way comparison like-for-like (previously only DQN/ComplexEnv was
    evaluated at its best checkpoint while the others used final weights).
    """

    def __init__(self, name, every=100, window=100, enabled=True):
        self.name = name
        self.every = int(every)
        self.window = int(window)
        self.enabled = bool(enabled)
        self.path = os.path.join(ARTIFACTS, "checkpoints", f"{name}_best.pt")
        self.best_score = None
        self.best_episode = None

    def _score(self, logger):
        sr = logger.get_success_rate(self.window)
        stage = logger.get_mean("stages", self.window) if logger.stages else 0.0
        shaped = logger.get_mean("rewards", self.window)
        return (round(sr, 6), round(stage, 6), round(shaped, 6))

    def maybe_update(self, agent, logger, episode, force=False):
        """Call once per episode with the 1-based global episode number."""
        if not self.enabled:
            return False
        if not force and episode % self.every != 0:
            return False
        score = self._score(logger)
        if self.best_score is not None and score <= self.best_score:
            return False
        self.best_score = score
        self.best_episode = int(episode)
        state = _agent_state(agent)
        state["episode"] = int(episode)
        state["score"] = list(score)
        torch.save(state, self.path)
        return True

    def load_best(self, agent, verbose=True):
        if not os.path.exists(self.path):
            if verbose:
                print(f"[{self.name}] no best checkpoint on disk — keeping final weights.")
            return agent
        state = torch.load(self.path, map_location=device)
        for attr in AGENT_MODULES:
            module = getattr(agent, attr, None)
            if module is not None and attr in state:
                module.load_state_dict(state[attr])
        if verbose:
            sr, stage, shaped = state.get("score", (float("nan"),) * 3)
            print(f"[{self.name}] restored best checkpoint from episode "
                  f"{state.get('episode')} (SR{self.window}={sr:.3f}, "
                  f"mean stage={stage:.2f}, mean shaped={shaped:+.2f})")
        return agent


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _rolling(data, window):
    """Simple rolling average via convolution."""
    data = list(data)
    if len(data) == 0:
        return np.array([])
    window = max(1, min(window, len(data)))
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def _plot_rolling(ax, data, window, color, label=None, raw_alpha=0.25):
    data = list(data)
    if not data:
        return
    if raw_alpha:
        ax.plot(data, alpha=raw_alpha, color=color)
    r = _rolling(data, window)
    x0 = max(0, len(data) - len(r))
    ax.plot(range(x0, x0 + len(r)), r, color=color, lw=2, label=label)


def _success_conditioned_steps(logger, window=50):
    """Rolling mean episode length over *successful* episodes only.

    Once the success rate plateaus this is the curve that still shows learning:
    it answers "is the agent reaching the goal faster?", which the unconditional
    steps-per-episode curve hides (failures always run to the step cap).
    """
    idx = [i for i, s in enumerate(logger.successes) if s > 0]
    lens = [logger.steps[i] for i in idx]
    if not lens:
        return [], np.array([])
    r = _rolling(lens, window)
    return idx[len(idx) - len(r):], r


def _has_shaped_discounted(logger):
    vals = getattr(logger, "disc_rewards", [])
    return any(v == v for v in vals)      # any non-NaN


def plot_training(logger, title="Training", window=50):
    """6-panel training curves for a single run.

    Row 1: undiscounted return, discounted return (the requested
           "discounted reward vs episode"), success rate (+ greedy probes).
    Row 2: steps per episode, steps per *successful* episode, cumulative
           environment steps (the common sample-budget axis).
    """
    fig, axes = plt.subplots(2, 3, figsize=(19, 9))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # --- undiscounted return -------------------------------------------------
    ax = axes[0, 0]
    _plot_rolling(ax, logger.rewards, window, "royalblue", label="shaped")
    _plot_rolling(ax, logger.sparse_rewards, window, "darkorange",
                  label="sparse (true env)", raw_alpha=0.0)
    ax.set(xlabel="Episode", ylabel="Return", title="Undiscounted Return per Episode")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- discounted return ---------------------------------------------------
    ax = axes[0, 1]
    g = getattr(logger, "gamma", 0.99)
    if _has_shaped_discounted(logger):
        _plot_rolling(ax, logger.disc_rewards, window, "royalblue", label="shaped")
    _plot_rolling(ax, logger.disc_sparse_rewards, window, "darkorange",
                  label="sparse (true env)", raw_alpha=0.0)
    ax.set(xlabel="Episode", ylabel=r"$G_0=\sum_t \gamma^t r_{t+1}$",
           title=f"Discounted Return per Episode ($\\gamma$={g})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- success rate --------------------------------------------------------
    ax = axes[0, 2]
    r = _rolling(logger.successes, window)
    x0 = max(0, len(logger.successes) - len(r))
    ax.plot(range(x0, x0 + len(r)), r, color="crimson", lw=2,
            label=f"rolling {window} (behaviour policy)")
    if getattr(logger, "probe_episodes", []):
        ax.plot(logger.probe_episodes, logger.probe_successes, "o--", color="black",
                ms=4, lw=1, label="greedy probe ($\\epsilon$=0)")
    ax.set(xlabel="Episode", ylabel="Success Rate", title="Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- steps ---------------------------------------------------------------
    ax = axes[1, 0]
    _plot_rolling(ax, logger.steps, window, "seagreen")
    ax.set(xlabel="Episode", ylabel="Steps", title="Steps per Episode (all episodes)")
    ax.grid(True, alpha=0.3)

    # --- steps, successes only ----------------------------------------------
    ax = axes[1, 1]
    xs, ys = _success_conditioned_steps(logger, window)
    if len(ys):
        ax.plot(xs, ys, color="seagreen", lw=2)
    else:
        ax.text(0.5, 0.5, "no successful episodes", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    ax.set(xlabel="Episode", ylabel="Steps",
           title="Steps per Successful Episode (goal-reaching speed)")
    ax.grid(True, alpha=0.3)

    # --- cumulative env steps ------------------------------------------------
    ax = axes[1, 2]
    ax.plot(logger.cumulative_steps, color="darkorchid", lw=2)
    ax.set(xlabel="Episode", ylabel="Cumulative Steps",
           title="Cumulative Environment Steps")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig


def plot_comparison(loggers, names, title="Algorithm Comparison", window=50):
    """Overlay the same six curves for multiple algorithms on a common axis."""
    fig, axes = plt.subplots(2, 3, figsize=(19, 9))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, (lg, name) in enumerate(zip(loggers, names)):
        c = colors[i % len(colors)]

        _plot_rolling(axes[0, 0], lg.rewards, window, c, label=name, raw_alpha=0.0)
        _plot_rolling(axes[0, 1], lg.disc_sparse_rewards, window, c, label=name,
                      raw_alpha=0.0)
        _plot_rolling(axes[0, 2], lg.successes, window, c, label=name, raw_alpha=0.0)
        _plot_rolling(axes[1, 0], lg.steps, window, c, label=name, raw_alpha=0.0)
        xs, ys = _success_conditioned_steps(lg, window)
        if len(ys):
            axes[1, 1].plot(xs, ys, color=c, lw=2, label=name)
        axes[1, 2].plot(lg.cumulative_steps, color=c, lw=2, label=name)

    titles = [
        ("Undiscounted Shaped Return", "Return"),
        ("Discounted Sparse Return $G_0$", r"$G_0^{sparse}$"),
        (f"Success Rate (rolling {window})", "Success Rate"),
        ("Steps per Episode", "Steps"),
        ("Steps per Successful Episode", "Steps"),
        ("Cumulative Environment Steps", "Cumulative Steps"),
    ]
    for ax, (t, ylab) in zip(axes.ravel(), titles):
        ax.set(xlabel="Episode", ylabel=ylab, title=t)
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    axes[0, 2].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.show()
    return fig


def plot_stage_progress(logger, title="ComplexEnv Stage Progress", window=50):
    """Stage-reached scatter + distribution for ComplexEnv runs."""
    if not logger.stages:
        print("No stage data to plot.")
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    ax = axes[0]
    data = logger.stages
    ax.scatter(range(len(data)), data, alpha=0.2, s=8, c="teal")
    r = _rolling(data, window)
    x0 = max(0, len(data) - len(r))
    ax.plot(range(x0, x0 + len(r)), r, color="crimson", lw=2)
    ax.set(xlabel="Episode", ylabel="Stage", title="Max Stage per Episode")
    ax.set_yticks(range(6))
    ax.set_yticklabels(STAGE_LABELS)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    counts = [logger.stages.count(i) for i in range(6)]
    bar_colors = ["#999", "#9b59b6", "#8B4513", "#3498db", "#e67e22", "#2ecc71"]
    ax.bar(STAGE_LABELS, counts, color=bar_colors)
    ax.set(xlabel="Stage", ylabel="Episodes", title="Stage Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()
    return fig


# Canonical stage names, used by every stage plot/table in the notebook.
STAGE_LABELS = ["None", "Key", "Door", "Water", "Lava", "Goal"]

print("Utilities loaded.")
