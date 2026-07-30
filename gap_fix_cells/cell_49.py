# ---------------------------------------------------------------------------
# Shared training helpers
# ---------------------------------------------------------------------------
def greedy_probe(env_factory, agent, n_episodes=5, seed=42, episode_offset=0):
    """Success rate of the *greedy* policy on held-out probe seeds.

    Training success is measured under the behaviour policy (epsilon-greedy for
    DQN, a sampled stochastic policy for REINFORCE/PPO) while the reported results
    are greedy. Probing periodically makes the training curve and the evaluation
    measure the same policy, which is what separates learning progress from
    exploration luck (e.g. a training SR100 of 0.33 at eps=0.15 alongside 0%
    greedy evaluation).
    """
    env = env_factory()
    successes = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=episode_seed(seed, episode_offset + ep, "probe"))
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.select_action(obs, training=False)
            obs, _, terminated, truncated, info = env.step(action)
        successes.append(float(info.get("is_success", terminated)))
    env.close()
    return float(np.mean(successes)) if successes else 0.0


class EarlyStopper:
    """Stop training when neither success rate nor stage progress improves.

    Stated criterion, applied identically wherever it is enabled: every episode,
    score = (rolling success rate over `window`, rolling mean stage). Training
    stops once `patience` consecutive episodes pass without the score improving by
    at least `min_delta`. This makes an unequal episode budget a measured decision
    with a reported stopping episode, rather than an undocumented one.
    """

    def __init__(self, patience=None, window=100, min_delta=0.01):
        self.patience = patience
        self.window = window
        self.min_delta = min_delta
        self.best = None
        self.since = 0
        self.stopped_at = None

    def __call__(self, logger, episode):
        if not self.patience:
            return False
        score = (logger.get_success_rate(self.window)
                 + 0.01 * (logger.get_mean("stages", self.window) if logger.stages else 0.0))
        if self.best is None or score > self.best + self.min_delta:
            self.best, self.since = score, 0
        else:
            self.since += 1
        if self.since >= self.patience:
            self.stopped_at = episode
            return True
        return False


def _record_mid_training_video(env_factory, agent, fname, global_ep):
    reward, info = record_agent_video(env_factory, agent, fname, seed=99)
    print(f"  >> mid-training video at ep {global_ep}: {os.path.basename(fname)} "
          f"(R={reward:+.2f}, success={bool(info.get('is_success', False))})")


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------
def train_dqn(
    env_factory,
    n_episodes,
    device=device,
    video_at=None,
    seed=42,
    base_seed=None,
    existing_agent=None,
    checkpointer=None,
    probe_every=None,
    probe_episodes=5,
    episode_offset=0,
    log_every=100,
    **agent_kw,
):
    """Train (or continue training) a DQN agent on the given env factory.

    Parameters
    ----------
    env_factory : callable
        Returns a fresh wrapped environment.
    n_episodes : int
        Number of episodes to run in this call.
    video_at : dict[int, str] | None
        {episode_number: filename} mid-training snapshots; episode numbers are
        relative to THIS call.
    seed / base_seed : int
        `seed` seeds the agent's RNG use; per-episode env reset seeds come from
        `base_seed` when given, else `seed`.
    existing_agent : DQNAgent | None
        Continue training this agent (net, target net, replay buffer, optimizer,
        step counter) instead of building a fresh one.
    checkpointer : BestCheckpointer | None
        Best-so-far model selection (identical rule for all three algorithms).
    probe_every / probe_episodes : int | None, int
        Run `probe_episodes` greedy (eps=0) episodes every `probe_every` episodes
        and log the result next to the behaviour-policy success rate.
    episode_offset : int
        Added to printed / checkpointed episode numbers when resuming, so logs are
        globally numbered instead of restarting at 1 in every chunk.
    **agent_kw
        Passed to DQNAgent(...) only when building a fresh agent.
    """
    env = env_factory()

    if existing_agent is None:
        in_ch = env.observation_space.shape[0]
        n_act = env.action_space.n
        agent = DQNAgent(in_ch, n_act, device=device, **agent_kw)
    else:
        agent = existing_agent

    logger = TrainingLogger(gamma=agent.gamma)
    reset_seed_base = seed if base_seed is None else base_seed

    for ep in range(n_episodes):
        obs, info = env.reset(seed=episode_seed(reset_seed_base, ep, "train"))
        ep_r, ep_sparse, ep_disc, ep_disc_sparse, ep_s = 0.0, 0.0, 0.0, 0.0, 0
        discount = 1.0

        while True:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # A time limit is not a terminal state: only `terminated` zeroes the
            # bootstrap in the Bellman target. `boundary` tells the buffer the
            # episode ended here so it stores this next-state explicitly.
            agent.buffer.push(obs, action, reward, next_obs,
                              float(terminated),
                              boundary=bool(terminated or truncated))
            agent.steps += 1

            if agent.steps % agent.train_freq == 0:
                agent.update()
            agent.maybe_sync_target()   # env-step clock, independent of train_freq

            obs = next_obs
            sparse_r = float(info.get("sparse_reward", 0.0))
            ep_r += reward
            ep_sparse += sparse_r
            ep_disc += discount * reward
            ep_disc_sparse += discount * sparse_r
            discount *= agent.gamma
            ep_s += 1

            if terminated or truncated:
                break

        success = info.get("is_success", terminated)
        logger.log(ep_r, ep_s, success, info, sparse_reward=ep_sparse,
                   disc_reward=ep_disc, disc_sparse_reward=ep_disc_sparse)
        global_ep = episode_offset + ep + 1

        if checkpointer is not None and checkpointer.maybe_update(
                agent, logger, global_ep, force=(ep + 1 == n_episodes)):
            print(f"  [best] DQN ep {global_ep}: SR100={logger.get_success_rate(100):.3f}"
                  f" -> {os.path.basename(checkpointer.path)}")

        if probe_every and global_ep % probe_every == 0:
            probe_sr = greedy_probe(env_factory, agent, probe_episodes,
                                    seed=seed, episode_offset=global_ep)
            logger.log_greedy_probe(len(logger.rewards) - 1, probe_sr)

        if video_at and (ep + 1) in video_at:
            _record_mid_training_video(env_factory, agent, video_at[ep + 1], global_ep)

        if global_ep % log_every == 0:
            print(f"  DQN ep {global_ep}/{episode_offset + n_episodes}  "
                  f"R={ep_r:+.2f}  G0={ep_disc:+.2f}  steps={ep_s}  "
                  f"eps={agent.get_epsilon():.3f}  "
                  f"SR100={logger.get_success_rate(100):.3f}")

    env.close()
    return agent, logger


def train_reinforce(
    env_factory,
    n_episodes,
    device=device,
    video_at=None,
    seed=42,
    checkpointer=None,
    probe_every=None,
    probe_episodes=5,
    early_stop_patience=None,
    log_every=100,
    **agent_kw,
):
    """Train a REINFORCE agent; returns (agent, logger).

    `early_stop_patience` enables the documented stopping rule in EarlyStopper.
    """
    env = env_factory()
    in_ch = env.observation_space.shape[0]
    n_act = env.action_space.n
    agent = REINFORCEAgent(in_ch, n_act, device=device, **agent_kw)
    logger = TrainingLogger(gamma=agent.gamma)
    stopper = EarlyStopper(early_stop_patience)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=episode_seed(seed, ep, "train"))
        ep_r, ep_sparse, ep_disc, ep_disc_sparse, ep_s = 0.0, 0.0, 0.0, 0.0, 0
        discount = 1.0

        while True:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.rewards.append(reward)
            obs = next_obs
            sparse_r = float(info.get("sparse_reward", 0.0))
            ep_r += reward
            ep_sparse += sparse_r
            ep_disc += discount * reward
            ep_disc_sparse += discount * sparse_r
            discount *= agent.gamma
            ep_s += 1
            if terminated or truncated:
                break

        # Bootstrap the truncated tail only when a learned baseline exists;
        # otherwise the finite-horizon bias is accepted and documented.
        agent.update(bootstrap_obs=obs if (truncated and not terminated) else None)

        success = info.get("is_success", terminated)
        logger.log(ep_r, ep_s, success, info, sparse_reward=ep_sparse,
                   disc_reward=ep_disc, disc_sparse_reward=ep_disc_sparse)
        global_ep = ep + 1

        if checkpointer is not None and checkpointer.maybe_update(
                agent, logger, global_ep, force=(ep + 1 == n_episodes)):
            print(f"  [best] REINFORCE ep {global_ep}: "
                  f"SR100={logger.get_success_rate(100):.3f}")

        if probe_every and global_ep % probe_every == 0:
            probe_sr = greedy_probe(env_factory, agent, probe_episodes,
                                    seed=seed, episode_offset=global_ep)
            logger.log_greedy_probe(len(logger.rewards) - 1, probe_sr)

        if video_at and global_ep in video_at:
            _record_mid_training_video(env_factory, agent, video_at[global_ep], global_ep)

        if global_ep % log_every == 0:
            print(f"  REINFORCE ep {global_ep}/{n_episodes}  R={ep_r:+.2f}  "
                  f"G0={ep_disc:+.2f}  steps={ep_s}  "
                  f"SR100={logger.get_success_rate(100):.3f}")

        if stopper(logger, global_ep):
            print(f"  [early stop] no improvement for {early_stop_patience} episodes "
                  f"-> stopping at episode {global_ep}/{n_episodes}")
            if checkpointer is not None:
                checkpointer.maybe_update(agent, logger, global_ep, force=True)
            break

    env.close()
    return agent, logger


def train_ppo(
    env_factory,
    n_episodes,
    device=device,
    video_at=None,
    seed=42,
    checkpointer=None,
    probe_every=None,
    probe_episodes=5,
    log_every=100,
    **agent_kw,
):
    """Train a PPO agent (step-based rollouts with episode counting).

    On an episode boundary the loop records whether it was a termination or a
    truncation and, for truncations, V(s_T) — so GAE can bootstrap through the
    time limit instead of treating it as the end of the world.
    """
    env = env_factory()
    in_ch = env.observation_space.shape[0]
    n_act = env.action_space.n
    agent = PPOAgent(in_ch, n_act, device=device, **agent_kw)
    logger = TrainingLogger(gamma=agent.gamma)

    obs, info = env.reset(seed=episode_seed(seed, 0, "train"))
    ep_r, ep_sparse, ep_disc, ep_disc_sparse, ep_s = 0.0, 0.0, 0.0, 0.0, 0
    ep_count = 0
    discount = 1.0

    while ep_count < n_episodes:
        for _ in range(agent.n_steps):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            time_limit = bool(truncated and not terminated)
            boundary_value = agent.get_value(next_obs) if time_limit else 0.0
            agent.buffer.add(obs, action, log_prob, reward, value,
                             terminated=float(terminated),
                             truncated=float(time_limit),
                             boundary_value=boundary_value)

            obs = next_obs
            sparse_r = float(info.get("sparse_reward", 0.0))
            ep_r += reward
            ep_sparse += sparse_r
            ep_disc += discount * reward
            ep_disc_sparse += discount * sparse_r
            discount *= agent.gamma
            ep_s += 1

            if terminated or truncated:
                success = info.get("is_success", terminated)
                logger.log(ep_r, ep_s, success, info, sparse_reward=ep_sparse,
                           disc_reward=ep_disc, disc_sparse_reward=ep_disc_sparse)
                ep_count += 1

                if checkpointer is not None and checkpointer.maybe_update(
                        agent, logger, ep_count, force=(ep_count == n_episodes)):
                    print(f"  [best] PPO ep {ep_count}: "
                          f"SR100={logger.get_success_rate(100):.3f}")

                if ep_count % log_every == 0:
                    d = agent.last_diagnostics
                    print(f"  PPO ep {ep_count}/{n_episodes}  "
                          f"shaped={ep_r:+.2f}  sparse={ep_sparse:+.2f}  "
                          f"G0={ep_disc:+.2f}  steps={ep_s}  "
                          f"SR100={logger.get_success_rate(100):.3f}  "
                          f"KL={d.get('approx_kl', float('nan')):.4f}  "
                          f"clip={d.get('clip_frac', float('nan')):.2f}  "
                          f"H={d.get('entropy', float('nan')):.3f}  "
                          f"EV={d.get('explained_variance', float('nan')):.2f}")

                if probe_every and ep_count % probe_every == 0:
                    probe_sr = greedy_probe(env_factory, agent, probe_episodes,
                                            seed=seed, episode_offset=ep_count)
                    logger.log_greedy_probe(len(logger.rewards) - 1, probe_sr)

                if video_at and ep_count in video_at:
                    _record_mid_training_video(env_factory, agent,
                                               video_at[ep_count], ep_count)

                ep_r, ep_sparse, ep_disc, ep_disc_sparse, ep_s = 0.0, 0.0, 0.0, 0.0, 0
                discount = 1.0
                obs, info = env.reset(seed=episode_seed(seed, ep_count, "train"))
                if ep_count >= n_episodes:
                    break

        if len(agent.buffer) > 0:
            agent.set_lr_fraction(1.0 - ep_count / max(1, n_episodes))
            agent.update(agent.get_value(obs))

    env.close()
    return agent, logger


# ============================================================
# Evaluation: greedy vs sampled policy modes
# ============================================================

def select_eval_action(agent, obs, device=device, mode="greedy"):
    """Select an action for evaluation.

    mode:
      - "greedy": deterministic argmax
      - "sample": stochastic sample from the policy distribution

    For DQN both modes use argmax, because DQN has no stochastic policy.
    """
    t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        # DQN
        if hasattr(agent, "q_net"):
            q = agent.q_net(t)
            return int(q.argmax(1).item())

        # REINFORCE
        if getattr(agent, "policy", None) is not None:
            logits = agent.policy(t)
        # PPO
        elif getattr(agent, "network", None) is not None:
            logits, _ = agent.network(t)
        else:
            raise ValueError("Unknown agent type")

        if mode == "sample":
            dist = torch.distributions.Categorical(logits=logits)
            return int(dist.sample().item())
        return int(logits.argmax(1).item())


def evaluate_agent_mode(env_factory, agent, n_episodes=50, seed=42, mode="greedy"):
    """Evaluate an agent with either greedy or sampled action selection.

    One environment is built per evaluation run and re-seeded per episode (the
    same idiom the training loops use), instead of constructing and closing a
    fresh env for each of the 50-200 episodes.

    Sampled evaluation is reproducible across reruns because the torch RNG is
    reseeded per episode from the evaluation seed and the episode index.
    """
    results = {
        "rewards": [],
        "sparse_rewards": [],
        "steps": [],
        "successes": [],
        "stages": [],
        "lava_deaths": [],
        "extinguished": [],
        "infos": [],
    }

    env = env_factory()
    for ep in range(n_episodes):
        obs, info = env.reset(seed=episode_seed(seed, ep, "eval"))

        if mode == "sample":
            eval_torch_seed = episode_seed(seed, ep, "eval_torch")
            torch.manual_seed(eval_torch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(eval_torch_seed)

        total_reward = 0.0
        total_sparse = 0.0
        ep_steps = 0

        while True:
            action = select_eval_action(agent, obs, device=device, mode=mode)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            total_sparse += float(info.get("sparse_reward", 0.0))
            ep_steps += 1

            if terminated or truncated:
                break

        results["rewards"].append(total_reward)
        results["sparse_rewards"].append(total_sparse)
        results["steps"].append(ep_steps)
        results["successes"].append(float(info.get("is_success", terminated)))
        results["stages"].append(int(info.get("stage_reached", 0)))
        results["lava_deaths"].append(int(bool(info.get("died_on_lava", False))))
        results["extinguished"].append(int(info.get("extinguished_lava_count", 0)))
        results["infos"].append(info)

    env.close()
    return results


def record_agent_video(env_factory, agent, filename, max_steps=500,
                       fps=10, seed=42):
    """Record a greedy rollout to mp4; returns (reward, info)."""
    env = env_factory()
    obs, info = env.reset(seed=episode_seed(seed, 0, "video"))
    frames = [env.render()]
    total_r = 0.0
    for _ in range(max_steps):
        action = agent.select_action(obs, training=False)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        total_r += reward
        if terminated or truncated:
            break
    imageio.mimsave(filename, frames, fps=fps)
    env.close()
    return total_r, info


print("Training & evaluation functions loaded.")
