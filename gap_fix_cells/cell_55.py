set_seed(42)
print("Training REINFORCE on SimpleRoomEnv ...")
t0 = time.time()

reinforce_simple_ckpt = BestCheckpointer("reinforce_simple")
reinforce_simple_agent, reinforce_simple_log = train_reinforce(
    make_simple_env,
    n_episodes=EPISODE_BUDGET["reinforce_simple"],
    seed=42,
    video_at={1500: video_path("reinforce_simple_mid1500.mp4")},
    checkpointer=reinforce_simple_ckpt,
    probe_every=PROBE_EVERY,
    probe_episodes=PROBE_EPISODES,
    **BEST_SETTINGS["reinforce_simple"],
)
print(f"Done in {time.time()-t0:.0f}s -- final SR100="
      f"{reinforce_simple_log.get_success_rate(100):.3f}")

# --- persist results BEFORE plotting/eval ---
save_logger(reinforce_simple_log, "reinforce_simple")
save_agent(reinforce_simple_agent, "reinforce_simple_final")

reinforce_simple_ckpt.load_best(reinforce_simple_agent)
save_agent(reinforce_simple_agent, "reinforce_simple")

# --- then plot / evaluate ---
fig = plot_training(reinforce_simple_log, title="REINFORCE -- SimpleRoomEnv")
save_figure(fig, "reinforce_simple_training")
