# =========================================================
# Reload agents + loggers from saved artifacts (no training)
# =========================================================
LOAD_FROM_ARTIFACTS = False      # set True to skip the training cells entirely

RUN_SPECS = {
    # name              (algorithm,   env factory,        settings key)
    "dqn_simple":       ("dqn",       "simple",  "dqn_simple"),
    "reinforce_simple": ("reinforce", "simple",  "reinforce_simple"),
    "ppo_simple":       ("ppo",       "simple",  "ppo_simple"),
    "dqn_complex":      ("dqn",       "complex", "dqn_complex"),
    "reinforce_complex": ("reinforce", "complex", "reinforce_complex"),
    "ppo_complex":      ("ppo",       "complex", "ppo_complex"),
}

ENV_FACTORIES = {"simple": make_simple_env, "complex": make_complex_env}

# Only the constructor arguments the network shapes depend on are needed to
# rebuild an agent for inference; the rest of BEST_SETTINGS is training-only.
_AGENT_CTOR_KEYS = {
    "dqn": ("gamma",),
    "reinforce": ("gamma", "baseline"),
    "ppo": ("gamma", "gae_lambda"),
}


def build_agent(algorithm, env_factory, settings_key, device=device):
    """Construct an untrained agent with the architecture used for `settings_key`."""
    env = env_factory()
    in_ch = env.observation_space.shape[0]
    n_act = env.action_space.n
    env.close()
    cfg = {k: v for k, v in BEST_SETTINGS[settings_key].items()
           if k in _AGENT_CTOR_KEYS[algorithm]}
    cls = {"dqn": DQNAgent, "reinforce": REINFORCEAgent, "ppo": PPOAgent}[algorithm]
    return cls(in_ch, n_act, device=device, **cfg)


def load_run(name, prefer_best=True, verbose=True):
    """Rebuild (agent, logger) for one run from artifacts on disk."""
    algorithm, env_key, settings_key = RUN_SPECS[name]
    agent = build_agent(algorithm, ENV_FACTORIES[env_key], settings_key)

    ckpt_dir = os.path.join(ARTIFACTS, "checkpoints")
    candidates = ([f"{name}_best", name, f"{name}_final"] if prefer_best
                  else [name, f"{name}_final", f"{name}_best"])
    loaded_from = None
    for stem in candidates:
        if os.path.exists(os.path.join(ckpt_dir, f"{stem}.pt")):
            load_agent(agent, stem)
            loaded_from = stem
            break

    logger = None
    if os.path.exists(os.path.join(ARTIFACTS, "logs", f"{name}.json")):
        logger = load_logger(name)

    if verbose:
        eps = "-" if logger is None else f"{len(logger.rewards)} episodes"
        print(f"  {name:<18} weights={loaded_from or 'MISSING':<22} log={eps}")
    return agent, logger


def show_saved_figures(pattern=""):
    """Display every saved figure whose filename contains `pattern`."""
    from IPython.display import Image
    plot_dir = os.path.join(ARTIFACTS, "plots")
    names = sorted(f for f in os.listdir(plot_dir)
                   if f.endswith(".png") and pattern in f)
    if not names:
        print(f"No saved figures matching {pattern!r} in {plot_dir}")
    for f in names:
        print(f)
        display(Image(filename=os.path.join(plot_dir, f)))


if LOAD_FROM_ARTIFACTS:
    print("Reloading agents and loggers from artifacts "
          f"({os.path.join(ARTIFACTS, 'checkpoints')}):")
    dqn_simple_agent, dqn_simple_log = load_run("dqn_simple")
    reinforce_simple_agent, reinforce_simple_log = load_run("reinforce_simple")
    ppo_simple_agent, ppo_simple_log = load_run("ppo_simple")
    dqn_complex_agent, dqn_complex_log = load_run("dqn_complex")
    reinforce_complex_agent, reinforce_complex_log = load_run("reinforce_complex")
    ppo_complex_agent, ppo_complex_log = load_run("ppo_complex")
    print("\nDone. Skip the training cells and continue from the comparison / "
          "evaluation / video cells.")
else:
    print("LOAD_FROM_ARTIFACTS = False -> training cells below will run normally.")
    print("Set it to True to rebuild every agent and logger from saved artifacts "
          "instead (no training).")
