set_seed(42)
print("Training DQN on SimpleRoomEnv ...")
t0 = time.time()

dqn_simple_ckpt = BestCheckpointer("dqn_simple")
dqn_simple_agent, dqn_simple_log = train_dqn(
    make_simple_env,
    n_episodes=EPISODE_BUDGET["dqn_simple"],
    seed=42,
    video_at={750: video_path("dqn_simple_mid750.mp4")},
    checkpointer=dqn_simple_ckpt,
    probe_every=PROBE_EVERY,
    probe_episodes=PROBE_EPISODES,
    **BEST_SETTINGS["dqn_simple"],
)
print(f"Done in {time.time()-t0:.0f}s -- final SR100="
      f"{dqn_simple_log.get_success_rate(100):.3f}  "
      f"(final eps={dqn_simple_agent.get_epsilon():.3f}, "
      f"target syncs={dqn_simple_agent.n_syncs})")

# --- persist results BEFORE plotting/eval ---
save_logger(dqn_simple_log, "dqn_simple")
save_agent(dqn_simple_agent, "dqn_simple_final")

# --- select the reported model: best-so-far checkpoint (same rule everywhere) ---
dqn_simple_ckpt.load_best(dqn_simple_agent)
save_agent(dqn_simple_agent, "dqn_simple")

# --- then plot / evaluate ---
fig = plot_training(dqn_simple_log, title="DQN -- SimpleRoomEnv")
save_figure(fig, "dqn_simple_training")
