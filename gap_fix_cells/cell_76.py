# ============================================================
# ComplexEnv evaluation: greedy + sampled policy modes
# ============================================================

complex_eval_results = {}

complex_eval_specs = [
    ("DQN", "greedy", dqn_complex_agent, make_complex_env),
    ("REINFORCE", "greedy", reinforce_complex_agent, make_complex_env),
    ("REINFORCE", "sample", reinforce_complex_agent, make_complex_env),
    ("PPO", "greedy", ppo_complex_agent, make_complex_env),
    ("PPO", "sample", ppo_complex_agent, make_complex_env),
]

N_EVAL_COMPLEX = INFERENCE_SETTINGS["n_eval_episodes"]["ComplexEnv"]
print(f"ComplexEnv evaluation: {N_EVAL_COMPLEX} episodes per algorithm/mode")

for algo, mode, agent, factory in complex_eval_specs:
    print(f"Evaluating {algo} / ComplexEnv / {mode} ...", end=" ")
    res = evaluate_agent_mode(factory, agent, n_episodes=N_EVAL_COMPLEX, mode=mode)
    complex_eval_results[(algo, mode)] = res

    print(
        f"SR={np.mean(res['successes']):.2f}  "
        f"sparse={np.mean(res['sparse_rewards']):.2f}  "
        f"shaped={np.mean(res['rewards']):.2f}  "
        f"steps={np.mean(res['steps']):.1f}  "
        f"mean_stage={np.mean(res['stages']):.2f}  "
        f"max_stage={max(res['stages']) if res['stages'] else 0}  "
        f"lava_death={100 * np.mean(res['lava_deaths']):.1f}%  "
        f"ext={np.mean(res['extinguished']):.2f}"
    )

print("\n" + "=" * 110)
print(f"ComplexEnv summary (n_episodes={N_EVAL_COMPLEX} per algorithm/mode)")
print("=" * 110)
print(f"{'Algorithm':<12} {'Mode':<8} {'Env':<10} "
      f"{'Sparse':>10} {'Shaped':>10} {'Steps':>8} {'Success %':>10} "
      f"{'MeanStage':>10} {'MaxStage':>9} {'LavaDeath %':>12} {'Avg Ext.':>10}")
print("-" * 110)

for (algo, mode), res in complex_eval_results.items():
    print(
        f"{algo:<12} {mode:<8} {'Complex':<10} "
        f"{np.mean(res['sparse_rewards']):>+10.3f} "
        f"{np.mean(res['rewards']):>+10.2f} "
        f"{np.mean(res['steps']):>8.1f} "
        f"{100 * np.mean(res['successes']):>9.1f}% "
        f"{np.mean(res['stages']):>10.2f} "
        f"{max(res['stages']) if res['stages'] else 0:>9} "
        f"{100 * np.mean(res['lava_deaths']):>11.1f}% "
        f"{np.mean(res['extinguished']):>10.2f}"
    )

print("=" * 110)
print("MaxStage is the single best episode of the run; MeanStage is the honest "
      "central tendency (full distribution in the breakdown below).")

complex_eval_dump = {
    f"{algo}_{mode}": {kk: vv for kk, vv in res.items() if kk != "infos"}
    for (algo, mode), res in complex_eval_results.items()
}

with open(os.path.join(ARTIFACTS, "logs", "complex_eval_results_modes.json"), "w") as f:
    json.dump(complex_eval_dump, f, indent=2)

print("Saved complex_eval_results_modes.json")
