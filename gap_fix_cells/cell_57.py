set_seed(42)
print("Training PPO on SimpleRoomEnv ...")
t0 = time.time()

ppo_simple_ckpt = BestCheckpointer("ppo_simple")
ppo_simple_agent, ppo_simple_log = train_ppo(
    make_simple_env,
    n_episodes=EPISODE_BUDGET["ppo_simple"],
    seed=42,
    video_at={1250: video_path("ppo_simple_mid1250.mp4")},
    checkpointer=ppo_simple_ckpt,
    probe_every=PROBE_EVERY,
    probe_episodes=PROBE_EPISODES,
    **BEST_SETTINGS["ppo_simple"],
)
print(f"Done in {time.time()-t0:.0f}s -- final SR100="
      f"{ppo_simple_log.get_success_rate(100):.3f}  "
      f"(last update: {ppo_simple_agent.last_diagnostics})")

# --- persist results BEFORE plotting/eval ---
save_logger(ppo_simple_log, "ppo_simple")
save_agent(ppo_simple_agent, "ppo_simple_final")

ppo_simple_ckpt.load_best(ppo_simple_agent)
save_agent(ppo_simple_agent, "ppo_simple")

# --- then plot / evaluate ---
fig = plot_training(ppo_simple_log, title="PPO -- SimpleRoomEnv")
save_figure(fig, "ppo_simple_training")
