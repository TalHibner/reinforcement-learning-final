# ============================================================
# SimpleRoomEnv evaluation: greedy + sampled policy modes
# ============================================================

simple_eval_results = {}

simple_eval_specs = [
    ("DQN", "greedy", dqn_simple_agent, make_simple_env),
    ("REINFORCE", "greedy", reinforce_simple_agent, make_simple_env),
    ("REINFORCE", "sample", reinforce_simple_agent, make_simple_env),
    ("PPO", "greedy", ppo_simple_agent, make_simple_env),
    ("PPO", "sample", ppo_simple_agent, make_simple_env),
]

N_EVAL_SIMPLE = INFERENCE_SETTINGS["n_eval_episodes"]["SimpleRoomEnv"]
print(f"SimpleRoomEnv evaluation: {N_EVAL_SIMPLE} episodes per algorithm/mode")

for algo, mode, agent, factory in simple_eval_specs:
    print(f"Evaluating {algo} / SimpleRoomEnv / {mode} ...", end=" ")
    res = evaluate_agent_mode(factory, agent, n_episodes=N_EVAL_SIMPLE, mode=mode)
    simple_eval_results[(algo, mode)] = res

    print(
        f"SR={np.mean(res['successes']):.2f}  "
        f"sparse={np.mean(res['sparse_rewards']):.2f}  "
        f"shaped={np.mean(res['rewards']):.2f}  "
        f"steps={np.mean(res['steps']):.1f}"
    )

print("\n" + "=" * 95)
print(f"SimpleRoomEnv summary (n_episodes={N_EVAL_SIMPLE} per algorithm/mode)")
print("=" * 95)
print(f"{'Algorithm':<12} {'Mode':<8} {'Env':<14} "
      f"{'Sparse':>10} {'Shaped':>10} {'Steps':>8} {'Success %':>10}")
print("-" * 95)

for (algo, mode), res in simple_eval_results.items():
    print(
        f"{algo:<12} {mode:<8} {'SimpleRoom':<14} "
        f"{np.mean(res['sparse_rewards']):>+10.3f} "
        f"{np.mean(res['rewards']):>+10.2f} "
        f"{np.mean(res['steps']):>8.1f} "
        f"{100 * np.mean(res['successes']):>9.1f}%"
    )

print("=" * 95)

simple_eval_dump = {
    f"{algo}_{mode}": {kk: vv for kk, vv in res.items() if kk != "infos"}
    for (algo, mode), res in simple_eval_results.items()
}

with open(os.path.join(ARTIFACTS, "logs", "simple_eval_results_modes.json"), "w") as f:
    json.dump(simple_eval_dump, f, indent=2)

print("Saved simple_eval_results_modes.json")
