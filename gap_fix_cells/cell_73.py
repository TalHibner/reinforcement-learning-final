fig = plot_comparison(
    [dqn_complex_log, reinforce_complex_log, ppo_complex_log],
    ["DQN", "REINFORCE", "PPO"],
    title="ComplexEnv -- Algorithm Comparison",
)
save_figure(fig, "complex_comparison")

# Stage comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
width = 0.25
x = np.arange(6)
for i, (lg, name) in enumerate(zip(
    [dqn_complex_log, reinforce_complex_log, ppo_complex_log],
    ["DQN", "REINFORCE", "PPO"])):
    counts = [lg.stages.count(s) for s in range(6)] if lg.stages else [0] * 6
    ax.bar(x + i * width, counts, width, label=name)
ax.set_xticks(x + width)
ax.set_xticklabels(STAGE_LABELS)
ax.set(xlabel="Max Stage Reached", ylabel="Episodes",
       title="ComplexEnv -- Stage Distribution by Algorithm (training)")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
save_figure(fig, "complex_training_stage_bar")
plt.show()

# Episode counts differ if REINFORCE early-stopped: state the budgets explicitly so
# the bar heights are read against the right denominators.
for lg, name in [(dqn_complex_log, "DQN"), (reinforce_complex_log, "REINFORCE"),
                 (ppo_complex_log, "PPO")]:
    print(f"{name:<10} {len(lg.rewards):>5} episodes, {lg.total_steps:>9,} env steps")
