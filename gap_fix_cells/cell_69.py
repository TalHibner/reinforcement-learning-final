set_seed(42)
print("Training REINFORCE on ComplexEnv ...")
t0 = time.time()

reinforce_complex_ckpt = BestCheckpointer("reinforce_complex")
reinforce_complex_agent, reinforce_complex_log = train_reinforce(
    make_complex_env,
    n_episodes=EPISODE_BUDGET["reinforce_complex"],
    seed=42,
    video_at={2000: video_path("reinforce_complex_mid2000.mp4")},
    checkpointer=reinforce_complex_ckpt,
    probe_every=PROBE_EVERY,
    probe_episodes=PROBE_EPISODES,
    early_stop_patience=REINFORCE_EARLY_STOP_PATIENCE,
    **BEST_SETTINGS["reinforce_complex"],
)
print(f"Done in {time.time()-t0:.0f}s -- "
      f"episodes run={len(reinforce_complex_log.rewards)}"
      f"/{EPISODE_BUDGET['reinforce_complex']}  "
      f"final SR100={reinforce_complex_log.get_success_rate(100):.3f}  "
      f"env steps={reinforce_complex_log.total_steps:,}")

# --- persist results BEFORE plotting/eval ---
save_logger(reinforce_complex_log, "reinforce_complex")
save_agent(reinforce_complex_agent, "reinforce_complex_final")

reinforce_complex_ckpt.load_best(reinforce_complex_agent)
save_agent(reinforce_complex_agent, "reinforce_complex")

# --- then plot / evaluate ---
fig = plot_training(reinforce_complex_log, title="REINFORCE -- ComplexEnv")
save_figure(fig, "reinforce_complex_training")

fig = plot_stage_progress(reinforce_complex_log,
                         title="REINFORCE -- ComplexEnv Stage Progress")
save_figure(fig, "reinforce_complex_stage_progress")
