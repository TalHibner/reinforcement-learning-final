# ============================================================
# ComplexEnv -- Evaluation Stage Distribution Grouped Bar Chart
# ============================================================

eval_stage_keys = [
    ("DQN", "greedy"),
    ("REINFORCE", "greedy"),
    ("REINFORCE", "sample"),
    ("PPO", "greedy"),
    ("PPO", "sample"),
]

eval_stage_names = [
    "DQN greedy",
    "REINFORCE greedy",
    "REINFORCE sample",
    "PPO greedy",
    "PPO sample",
]

stage_counts = []

for key in eval_stage_keys:
    res = complex_eval_results[key]

    if "stages" in res and len(res["stages"]) > 0:
        stages = [int(s) for s in res["stages"]]
    else:
        stages = [int(info.get("stage_reached", 0)) for info in res["infos"]]

    counts = [stages.count(s) for s in range(6)]
    stage_counts.append(counts)

stage_counts = np.array(stage_counts)

fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(STAGE_LABELS))
width = 0.15

for i, name in enumerate(eval_stage_names):
    ax.bar(
        x + (i - 2) * width,
        stage_counts[i],
        width,
        label=name,
    )

ax.set_xticks(x)
ax.set_xticklabels(STAGE_LABELS)
ax.set(
    xlabel="Max Stage Reached",
    ylabel="Evaluation Episodes",
    title="ComplexEnv -- Evaluation Stage Distribution by Algorithm and Mode",
)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
save_figure(fig, "complex_eval_stage_bar_modes")
plt.show()
