# =========================================================
# Sanity checks & unit tests for the training machinery
# Every assertion below targets a bug that was actually present in an earlier
# revision of this notebook, so the tests double as regression tests.
# =========================================================

# --- Check 1: fresh training runs and returns a usable agent ---------------
tmp_agent, tmp_log = train_dqn(
    make_simple_env,
    n_episodes=10,
    seed=42,
    lr=1e-4,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.10,
    eps_decay_steps=100,
    buffer_size=100,
    batch_size=4,
    target_update=10,
    train_freq=4,
    log_every=1000,
)
print("[check 1] rewards logged:", len(tmp_log.rewards))
print("[check 1] agent has q_net:", hasattr(tmp_agent, "q_net"))
assert len(tmp_log.rewards) == len(tmp_log.cumulative_steps) == 10
assert tmp_log.cumulative_steps[-1] == sum(tmp_log.steps)

# --- Check 2: resume works via existing_agent ------------------------------
tmp_agent2, tmp_log2 = train_dqn(
    make_simple_env,
    n_episodes=5,
    seed=42,
    existing_agent=tmp_agent,
    episode_offset=10,
    log_every=1000,
)
print("[check 2] extra rewards logged:", len(tmp_log2.rewards))
print("[check 2] agent object reused:", tmp_agent2 is tmp_agent)

# --- Check 3: reset determinism when seeded --------------------------------
env_probe = make_simple_env()
obs1, _ = env_probe.reset(seed=episode_seed(42, 0, "train"))
obs2, _ = env_probe.reset(seed=episode_seed(42, 0, "train"))
print("[check 3] deterministic reset:", np.array_equal(obs1, obs2))
assert np.array_equal(obs1, obs2)
env_probe.close()

# --- Check 4: target sync runs on the env-step clock, not train_freq's -----
# (M4) target_update=10 with train_freq=4 must sync every 10 env steps, not 20.
probe_agent = DQNAgent(1, 3, target_update=10, train_freq=4, device="cpu")
for _ in range(100):
    probe_agent.steps += 1
    probe_agent.maybe_sync_target()
print(f"[check 4] syncs in 100 env steps with target_update=10, train_freq=4: "
      f"{probe_agent.n_syncs} (expected 10)")
assert probe_agent.n_syncs == 10

# --- Check 5: replay buffer next-state linkage ------------------------------
# Observations encode their own step index, so a sampled transition can be
# checked against the reference sequence. Covers the "next_state[t] is
# state[t+1]" shortcut, the explicitly stored boundary next-states, and the
# uint8 round-trip.
buf = ReplayBuffer(capacity=16)
ref = {}
step_idx = 0
for episode in range(6):
    for t in range(5):
        s = np.full((1, 2, 2), step_idx / 255.0, dtype=np.float32)
        s2 = np.full((1, 2, 2), (step_idx + 1) / 255.0, dtype=np.float32)
        boundary = (t == 4)
        buf.push(s, step_idx % 3, float(step_idx), s2,
                 float(boundary and t == 4), boundary=boundary)
        ref[step_idx] = (step_idx + 1, step_idx % 3, float(step_idx))
        step_idx += 1

for _ in range(200):
    bs, ba, br, bs2, bd = buf.sample(8)
    for i in range(8):
        got_s = int(round(float(bs[i].flat[0]) * 255))
        got_s2 = int(round(float(bs2[i].flat[0]) * 255))
        exp_s2, exp_a, exp_r = ref[got_s]
        assert got_s2 == exp_s2, f"next_state mismatch at {got_s}: {got_s2} != {exp_s2}"
        assert int(ba[i]) == exp_a and float(br[i]) == exp_r
print(f"[check 5] replay buffer: 1600 sampled transitions verified "
      f"(len={len(buf)}, boundary next-states stored={len(buf.boundary_next)})")

roundtrip = ReplayBuffer._to_float(ReplayBuffer._to_storage(np.arange(256) / 255.0))
assert np.array_equal((roundtrip * 255).round().astype(np.int64), np.arange(256))
print("[check 5] uint8 round-trip exact for all 256 pixel values")

# --- Check 6: GAE treats truncation and termination differently -------------
# (M1) Hand-computed targets on a 3-step rollout.
gae_agent = PPOAgent(1, 3, gamma=0.9, gae_lambda=0.5, device="cpu")


def _gae_case(terminated, truncated, boundary_value):
    gae_agent.buffer.clear()
    # two ordinary steps, then a boundary step
    gae_agent.buffer.add(np.zeros((1, 84, 84), np.float32), 0, 0.0, 1.0, 2.0)
    gae_agent.buffer.add(np.zeros((1, 84, 84), np.float32), 0, 0.0, 1.0, 3.0)
    gae_agent.buffer.add(np.zeros((1, 84, 84), np.float32), 0, 0.0, 1.0, 4.0,
                         terminated=terminated, truncated=truncated,
                         boundary_value=boundary_value)
    adv, ret = gae_agent.compute_gae(next_value=7.0)
    gae_agent.buffer.clear()
    return [float(a) for a in adv], [float(r) for r in ret]


g, lam = 0.9, 0.5
# terminated: delta_2 = 1 + g*0 - 4 = -3
d2_term = 1 + g * 0.0 - 4.0
# truncated with V(s_T) = 10: delta_2 = 1 + g*10 - 4
d2_trunc = 1 + g * 10.0 - 4.0
d1 = 1 + g * 4.0 - 3.0
d0 = 1 + g * 3.0 - 2.0

adv_term, _ = _gae_case(terminated=1.0, truncated=0.0, boundary_value=0.0)
adv_trunc, _ = _gae_case(terminated=0.0, truncated=1.0, boundary_value=10.0)

exp_term = [d0 + g * lam * (d1 + g * lam * d2_term), d1 + g * lam * d2_term, d2_term]
exp_trunc = [d0 + g * lam * (d1 + g * lam * d2_trunc), d1 + g * lam * d2_trunc, d2_trunc]

print(f"[check 6] terminated advantages {np.round(adv_term, 4)} "
      f"== {np.round(exp_term, 4)}")
print(f"[check 6] truncated  advantages {np.round(adv_trunc, 4)} "
      f"== {np.round(exp_trunc, 4)}")
assert np.allclose(adv_term, exp_term, atol=1e-5)
assert np.allclose(adv_trunc, exp_trunc, atol=1e-5)
assert adv_term[2] != adv_trunc[2], "truncation must bootstrap, termination must not"

# --- Check 7: REINFORCE loss does not scale with episode length -------------
# (M2) With mean-reduced terms, an episode of 100 identical steps must produce
# the same policy loss as one of 10 — with the old SUM it was 10x larger, which
# is why the entropy bonus was effectively coef/T.
def _reinforce_policy_loss(n_steps):
    set_seed(0)
    agent = REINFORCEAgent(1, 3, gamma=0.0, baseline="none",
                           entropy_coef=0.01, device="cpu")
    obs = np.zeros((1, 84, 84), dtype=np.float32)
    agent._states = [obs] * n_steps
    agent._actions = [1] * n_steps
    agent.rewards = [-1.0] * n_steps
    agent.update()
    return agent.last_diagnostics["policy_loss"]


l10, l100 = _reinforce_policy_loss(10), _reinforce_policy_loss(100)
print(f"[check 7] policy loss T=10: {l10:.6f}   T=100: {l100:.6f}   "
      f"ratio {l100 / l10:.3f} (expected ~1.0)")
assert abs(l100 / l10 - 1.0) < 1e-3

# --- Check 8: legacy logs are repaired on load ------------------------------
# (C2/C8) Older saved logs have no discounted returns, and chunk-merged logs had
# a cumulative_steps array covering only the last chunk.
legacy = TrainingLogger(gamma=0.99)
for i in range(5):
    legacy.log(1.0, 10 + i, i % 2 == 0)
legacy.cumulative_steps = [10]          # simulate the old merge bug
legacy.disc_rewards = []
legacy.disc_sparse_rewards = []
save_logger(legacy, "_unit_test_legacy")
repaired = load_logger("_unit_test_legacy")
assert repaired.cumulative_steps == [int(x) for x in np.cumsum(repaired.steps)]
assert len(repaired.disc_sparse_rewards) == 5
assert abs(repaired.disc_sparse_rewards[0] - 0.99 ** 9) < 1e-9
print("[check 8] legacy log repaired: cumulative_steps rebuilt, "
      f"G0_sparse reconstructed ({repaired.disc_sparse_rewards[0]:.4f} = 0.99^9)")
os.remove(os.path.join(ARTIFACTS, "logs", "_unit_test_legacy.json"))

del tmp_agent, tmp_agent2, tmp_log, tmp_log2, probe_agent, gae_agent, buf
print("\nAll sanity checks passed.")
