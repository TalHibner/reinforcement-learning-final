# =========================================================
# DQN on ComplexEnv, with best-so-far checkpointing and greedy probes
# =========================================================
set_seed(42)
print("Training DQN on ComplexEnv ...")
t0 = time.time()

# Best-so-far checkpointing now happens inside the training loop (BestCheckpointer),
# so this run is a single call: no 200-episode chunking, no per-chunk seed bases,
# and episode numbers in the log are global.
dqn_complex_ckpt = BestCheckpointer("dqn_complex")
dqn_complex_agent, dqn_complex_log = train_dqn(
    make_complex_env,
    n_episodes=EPISODE_BUDGET["dqn_complex"],
    seed=42,
    video_at={2000: video_path("dqn_complex_mid2000.mp4")},
    checkpointer=dqn_complex_ckpt,
    probe_every=PROBE_EVERY,
    probe_episodes=PROBE_EPISODES,
    **BEST_SETTINGS["dqn_complex"],
)
print(f"Done in {time.time()-t0:.0f}s -- final SR100="
      f"{dqn_complex_log.get_success_rate(100):.3f}  "
      f"(final eps={dqn_complex_agent.get_epsilon():.3f}, "
      f"env steps={dqn_complex_log.total_steps:,}, "
      f"target syncs={dqn_complex_agent.n_syncs})")

# --- persist results BEFORE plotting/eval ---
save_logger(dqn_complex_log, "dqn_complex")
save_agent(dqn_complex_agent, "dqn_complex_final")

# --- select the reported model: best-so-far checkpoint ---
dqn_complex_ckpt.load_best(dqn_complex_agent)
save_agent(dqn_complex_agent, "dqn_complex")

# --- plots ---
fig = plot_training(dqn_complex_log, title="DQN -- ComplexEnv")
save_figure(fig, "dqn_complex_training")

fig = plot_stage_progress(dqn_complex_log, title="DQN -- ComplexEnv Stage Progress")
save_figure(fig, "dqn_complex_stage_progress")
