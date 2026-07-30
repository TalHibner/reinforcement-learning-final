# Display mid-training videos (recorded inline during training above).
# The labels below state the episode each clip was recorded at; the filenames are
# exactly the ones the training cells write via video_at={...}.
print("=" * 60)
print("MID-TRAINING SNAPSHOTS (recorded during actual training)")
print("=" * 60)

mid_videos = [
    ("dqn_simple_mid750.mp4",         "DQN -- SimpleRoomEnv (ep 750/2000)"),
    ("reinforce_simple_mid1500.mp4",  "REINFORCE -- SimpleRoomEnv (ep 1500/3000)"),
    ("ppo_simple_mid1250.mp4",        "PPO -- SimpleRoomEnv (ep 1250/2500)"),
    ("dqn_complex_mid2000.mp4",       "DQN -- ComplexEnv (ep 2000/4000)"),
    ("reinforce_complex_mid2000.mp4", "REINFORCE -- ComplexEnv (ep 2000/4000)"),
    ("ppo_complex_mid2000.mp4",       "PPO -- ComplexEnv (ep 2000/4000)"),
]
missing = []
for vname, label in mid_videos:
    fname = video_path(vname)
    if os.path.exists(fname):
        print(f"\n{label}")
        display(embed_mp4(fname))
    else:
        missing.append(label)
        print(f"  {label}: NOT FOUND at {fname}")

if missing:
    print(f"\n{len(missing)} of {len(mid_videos)} mid-training clips missing — "
          "re-run the corresponding training cell (or the REINFORCE-ComplexEnv run "
          "stopped early, before its snapshot episode).")

# --- Fully converged videos ---
print()
print("=" * 60)
print("CONVERGED VIDEOS (selected checkpoint, greedy evaluation)")
print("Note: These videos use a single fixed seed (99) for comparability.")
print("A single episode may not reflect the agent's average evaluation performance.")
print("=" * 60)

for name, agent, factory, label in [
    ("dqn_simple",       dqn_simple_agent,       make_simple_env,  "DQN -- SimpleRoomEnv"),
    ("reinforce_simple", reinforce_simple_agent, make_simple_env,  "REINFORCE -- SimpleRoomEnv"),
    ("ppo_simple",       ppo_simple_agent,       make_simple_env,  "PPO -- SimpleRoomEnv"),
    ("dqn_complex",       dqn_complex_agent,       make_complex_env, "DQN -- ComplexEnv"),
    ("reinforce_complex", reinforce_complex_agent, make_complex_env, "REINFORCE -- ComplexEnv"),
    ("ppo_complex",       ppo_complex_agent,       make_complex_env, "PPO -- ComplexEnv"),
]:
    fname = video_path(f"{name}_eval.mp4")
    r, info = record_agent_video(factory, agent, fname, seed=99)
    success = info.get("is_success", "N/A")
    print(f"\n{label}: reward={r:+.2f}  success={success}")
    display(embed_mp4(fname))
