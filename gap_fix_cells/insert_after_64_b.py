# ============================================================
# REINFORCE baseline ablation (SimpleRoomEnv, matched budget + seed)
# ============================================================
RUN_BASELINE_ABLATION = True
ABLATION_EPISODES = 800
ABLATION_EVAL_EPISODES = 50

ablation_logs, ablation_rows = {}, []

if RUN_BASELINE_ABLATION:
    for variant in ("episode", "running", "value"):
        set_seed(42)                        # identical init and env seeds per variant
        print(f"\n--- REINFORCE baseline='{variant}' "
              f"({ABLATION_EPISODES} episodes) ---")
        t0 = time.time()
        cfg = dict(BEST_SETTINGS["reinforce_simple"])
        cfg["baseline"] = variant
        agent, log = train_reinforce(
            make_simple_env, n_episodes=ABLATION_EPISODES, seed=42,
            log_every=200, **cfg,
        )
        ablation_logs[variant] = log

        greedy = evaluate_agent_mode(make_simple_env, agent,
                                     n_episodes=ABLATION_EVAL_EPISODES, mode="greedy")
        sampled = evaluate_agent_mode(make_simple_env, agent,
                                      n_episodes=ABLATION_EVAL_EPISODES, mode="sample")
        ablation_rows.append({
            "variant": variant,
            "train_SR100": log.get_success_rate(100),
            "greedy_SR": float(np.mean(greedy["successes"])),
            "sampled_SR": float(np.mean(sampled["successes"])),
            "mean_G0_sparse": float(np.mean(log.disc_sparse_rewards[-100:])),
            "entropy": agent.last_diagnostics.get("entropy", float("nan")),
            "seconds": time.time() - t0,
        })
        save_logger(log, f"reinforce_simple_ablation_{variant}")
        save_agent(agent, f"reinforce_simple_ablation_{variant}")
        del agent

    print("\n" + "=" * 96)
    print(f"REINFORCE baseline ablation — SimpleRoomEnv, {ABLATION_EPISODES} episodes, "
          f"eval n={ABLATION_EVAL_EPISODES}")
    print("=" * 96)
    print(f"{'baseline':<12}{'train SR100':>13}{'greedy SR':>11}{'sampled SR':>12}"
          f"{'mean G0 sparse':>16}{'final entropy':>15}{'seconds':>9}")
    print("-" * 96)
    for r in ablation_rows:
        print(f"{r['variant']:<12}{r['train_SR100']:>13.3f}{r['greedy_SR']:>11.3f}"
              f"{r['sampled_SR']:>12.3f}{r['mean_G0_sparse']:>16.4f}"
              f"{r['entropy']:>15.3f}{r['seconds']:>9.0f}")
    print("=" * 96)
    print("'episode' = the original per-episode whitening; 'running' = the default used "
          "for the reported runs; 'value' = learned V(s) baseline.")

    with open(os.path.join(ARTIFACTS, "logs", "reinforce_baseline_ablation.json"), "w") as f:
        json.dump(ablation_rows, f, indent=2)

    fig = plot_comparison(
        [ablation_logs[v] for v in ("episode", "running", "value")],
        ["baseline=episode (original)", "baseline=running (used)", "baseline=value"],
        title=f"REINFORCE Baseline Ablation -- SimpleRoomEnv ({ABLATION_EPISODES} episodes)",
    )
    save_figure(fig, "reinforce_baseline_ablation")
else:
    print("RUN_BASELINE_ABLATION = False -> ablation skipped "
          "(set True to reproduce the table referenced in the Discussion).")
