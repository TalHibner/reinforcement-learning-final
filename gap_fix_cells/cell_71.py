set_seed(42)
print("Training PPO on ComplexEnv ...")
t0 = time.time()

ppo_complex_ckpt = BestCheckpointer("ppo_complex")
ppo_complex_agent, ppo_complex_log = train_ppo(
    make_complex_env,
    n_episodes=EPISODE_BUDGET["ppo_complex"],
    seed=42,
    video_at={2000: video_path("ppo_complex_mid2000.mp4")},
    checkpointer=ppo_complex_ckpt,
    probe_every=PROBE_EVERY,
    probe_episodes=PROBE_EPISODES,
    **BEST_SETTINGS["ppo_complex"],
)
print(f"Done in {time.time()-t0:.0f}s -- final SR100="
      f"{ppo_complex_log.get_success_rate(100):.3f}  "
      f"env steps={ppo_complex_log.total_steps:,}  "
      f"(last update: {ppo_complex_agent.last_diagnostics})")

# --- persist results BEFORE plotting/eval ---
save_logger(ppo_complex_log, "ppo_complex")
save_agent(ppo_complex_agent, "ppo_complex_final")

ppo_complex_ckpt.load_best(ppo_complex_agent)
save_agent(ppo_complex_agent, "ppo_complex")

# --- then plot / evaluate ---
fig = plot_training(ppo_complex_log, title="PPO -- ComplexEnv")
save_figure(fig, "ppo_complex_training")

fig = plot_stage_progress(ppo_complex_log, title="PPO -- ComplexEnv Stage Progress")
save_figure(fig, "ppo_complex_stage_progress")
