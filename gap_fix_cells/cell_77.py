# ============================================================
# ComplexEnv -- Evaluation Stage Breakdown
# ============================================================

print("\n" + "=" * 100)
print(f"ComplexEnv -- Evaluation Stage Breakdown ({N_EVAL_COMPLEX} episodes)")
print("-" * 100)

for key in [
    ("DQN", "greedy"),
    ("REINFORCE", "greedy"),
    ("REINFORCE", "sample"),
    ("PPO", "greedy"),
    ("PPO", "sample"),
]:
    algo, mode = key
    res = complex_eval_results[key]

    if "stages" in res and len(res["stages"]) > 0:
        stages = [int(s) for s in res["stages"]]
    else:
        stages = [int(info.get("stage_reached", 0)) for info in res["infos"]]

    counts = {s: stages.count(s) for s in range(6)}
    dist_str = "  ".join(f"{STAGE_LABELS[s]}:{counts[s]}" for s in range(6))
    print(f"  {algo:<12} {mode:<8} {dist_str}    "
          f"(mean {np.mean(stages):.2f}, median {np.median(stages):.1f})")

print("=" * 100)
