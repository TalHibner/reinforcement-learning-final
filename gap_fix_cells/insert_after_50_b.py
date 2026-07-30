# =========================================================
# BEST SETTINGS — training + inference (single source of truth)
# =========================================================

# --- Training budgets (episodes). ComplexEnv budgets are equal across the three
# --- algorithms; REINFORCE additionally carries a documented early-stop rule.
EPISODE_BUDGET = {
    "dqn_simple": 2000, "reinforce_simple": 3000, "ppo_simple": 2500,
    "dqn_complex": 4000, "reinforce_complex": 4000, "ppo_complex": 4000,
}

# --- Per-(algorithm, env) training hyperparameters -------------------------
BEST_SETTINGS = {
    # ---------------- SimpleRoomEnv (3 actions, grayscale) ----------------
    "dqn_simple": dict(
        lr=1e-4, gamma=0.99,
        eps_start=1.0, eps_end=0.05, eps_decay_steps=60_000,   # <= run length
        buffer_size=30_000, batch_size=64,
        target_update=500, train_freq=4,
    ),
    "reinforce_simple": dict(
        lr=5e-5, gamma=0.99,
        entropy_coef=0.01,          # meaningful now that both loss terms are means
        baseline="running",
    ),
    "ppo_simple": dict(
        lr=1e-4, gamma=0.99, gae_lambda=0.95,
        clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01,
        n_epochs=4, n_steps=512, n_minibatches=4,
        target_kl=0.015, anneal_lr=True,
    ),
    # ---------------- ComplexEnv (6 actions, RGB) -------------------------
    "dqn_complex": dict(
        lr=1e-4, gamma=0.99,
        eps_start=1.0, eps_end=0.05, eps_decay_steps=600_000,
        buffer_size=100_000, batch_size=64,
        target_update=1000, train_freq=4,
    ),
    "reinforce_complex": dict(
        lr=5e-5, gamma=0.99, entropy_coef=0.02, baseline="running",
    ),
    "ppo_complex": dict(
        lr=1e-4, gamma=0.99, gae_lambda=0.95,
        clip_ratio=0.2, value_coef=0.5, entropy_coef=0.015,
        n_epochs=4, n_steps=1024, n_minibatches=4,
        target_kl=0.015, anneal_lr=True,
    ),
}

# --- Inference settings ----------------------------------------------------
INFERENCE_SETTINGS = dict(
    dqn="greedy argmax over Q(s, .); no exploration noise (eps = 0)",
    reinforce="greedy argmax over policy logits; sampled mode also reported",
    ppo="greedy argmax over policy logits; sampled mode also reported",
    n_eval_episodes={"SimpleRoomEnv": 200, "ComplexEnv": 50},
    env_seeds="fresh eval seed space (episode_seed(..., 'eval')), disjoint from training",
    torch_rng="reseeded per episode in sampled mode so sampled eval is reproducible",
)

# --- Model selection + probing (identical rule for all three algorithms) ---
MODEL_SELECTION = dict(
    rule="best-so-far checkpoint by rolling SR100, tie-broken by mean stage then "
         "mean shaped return; evaluated every 100 episodes (BestCheckpointer)",
    applies_to="DQN, REINFORCE and PPO, on both environments",
    greedy_probe="5 greedy (eps=0) episodes every 250 episodes, logged separately "
                 "from the behaviour-policy success rate",
)

# --- Seeding protocol ------------------------------------------------------
SEED_PROTOCOL = dict(
    master_seed=42,
    set_seed="Python / NumPy / torch (+ cuDNN deterministic) at the top of every run",
    per_episode="env.reset(seed=episode_seed(42, episode_idx, phase))",
    phases="train=+0, eval=+10M, video=+20M, probe=+30M, eval_torch=+40M (disjoint)",
)

PROBE_EVERY = 250
PROBE_EPISODES = 5
REINFORCE_EARLY_STOP_PATIENCE = 1000    # documented rule, see EarlyStopper


def _fmt(d, indent=2):
    pad = " " * indent
    return "\n".join(f"{pad}{k:<18} {v}" for k, v in d.items())


print("=" * 78)
print("BEST TRAINING SETTINGS")
print("=" * 78)
for name, cfg in BEST_SETTINGS.items():
    print(f"\n{name}  ({EPISODE_BUDGET[name]} episodes)")
    print(_fmt(cfg))

print("\n" + "=" * 78)
print("BEST INFERENCE SETTINGS")
print("=" * 78)
print(_fmt(INFERENCE_SETTINGS))

print("\n" + "=" * 78)
print("MODEL SELECTION & PROBING")
print("=" * 78)
print(_fmt(MODEL_SELECTION))
print(f"  {'probe_every':<18} {PROBE_EVERY} episodes x {PROBE_EPISODES} greedy episodes")
print(f"  {'early stop':<18} REINFORCE only, patience="
      f"{REINFORCE_EARLY_STOP_PATIENCE} episodes")

print("\n" + "=" * 78)
print("SEEDING PROTOCOL")
print("=" * 78)
print(_fmt(SEED_PROTOCOL))

print("\n" + "=" * 78)
print("ENVIRONMENT / PREPROCESSING")
print("=" * 78)
print("  SimpleRoomEnv    max_steps=100, grayscale 1x84x84, actions (left, right, "
      "forward), shaping: goal x10, step penalty 0.01")
print("  ComplexEnv       max_steps=400, RGB 3x84x84, actions (left, right, forward, "
      "pickup, drop, toggle)")
print("  ComplexEnv shaping " + ", ".join(f"{k}={v}" for k, v in COMPLEX_SHAPING.items()))
print("  Frame stacking   none (single frame is Markov here — see the architecture "
      "section for the argument)")
print("=" * 78)
