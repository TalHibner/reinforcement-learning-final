# ============================================================
# Auto-generated results tables (Markdown, straight from the eval dicts)
# ============================================================
EVAL_GAMMA = 0.99


def _disc_sparse(res, gamma=EVAL_GAMMA):
    """G0 of the true env reward per episode: success * gamma^(steps-1)."""
    return float(np.mean([s * gamma ** max(0, st - 1)
                          for s, st in zip(res["successes"], res["steps"])]))


def _stage_stats(res):
    st = np.asarray(res["stages"], dtype=float)
    if st.size == 0:
        return 0.0, 0.0, 0
    return float(st.mean()), float(np.median(st)), int(st.max())


def _md_table(header, rows):
    align = "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"
    out = ["| " + " | ".join(header) + " |", align]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def simple_results_markdown(results, n_eval):
    rows = []
    for (algo, mode), res in results.items():
        rows.append([
            algo, mode,
            f"{100 * np.mean(res['successes']):.1f}",
            f"{np.mean(res['sparse_rewards']):+.3f}",
            f"{_disc_sparse(res):+.4f}",
            f"{np.mean(res['rewards']):+.2f}",
            f"{np.mean(res['steps']):.1f}",
        ])
    header = ["Algorithm", "Mode", "Success %", "Avg sparse",
              "Avg $G_0$ sparse", "Avg shaped", "Avg steps"]
    return (f"**SimpleRoomEnv — evaluation, n={n_eval} episodes per algorithm/mode**\n\n"
            + _md_table(header, rows))


def complex_results_markdown(results, n_eval):
    rows = []
    for (algo, mode), res in results.items():
        mean_st, med_st, max_st = _stage_stats(res)
        rows.append([
            algo, mode,
            f"{100 * np.mean(res['successes']):.1f}",
            f"{np.mean(res['sparse_rewards']):+.3f}",
            f"{_disc_sparse(res):+.4f}",
            f"{np.mean(res['rewards']):+.2f}",
            f"{np.mean(res['steps']):.1f}",
            f"{mean_st:.2f}",
            f"{med_st:.1f}",
            f"{max_st} ({STAGE_LABELS[max_st]})",
            f"{100 * np.mean(res['lava_deaths']):.1f}",
            f"{np.mean(res['extinguished']):.2f}",
        ])
    header = ["Algorithm", "Mode", "Success %", "Avg sparse", "Avg $G_0$ sparse",
              "Avg shaped", "Avg steps", "Mean stage", "Median stage",
              "Max stage", "Lava death %", "Avg ext."]
    note = (f"**ComplexEnv — evaluation, n={n_eval} episodes per algorithm/mode.** "
            f"One episode = {100 / n_eval:.0f} percentage points, so single-episode "
            f"events (Avg ext. = {1 / n_eval:.2f}) are anecdotes, not rates.\n\n")
    return note + _md_table(header, rows)


display(Markdown(simple_results_markdown(simple_eval_results, 200)))
display(Markdown(complex_results_markdown(complex_eval_results, 50)))

# Training-budget summary — the common axis the cumulative-steps curves plot.
budget_rows = []
for name, log in [("DQN", dqn_complex_log), ("REINFORCE", reinforce_complex_log),
                  ("PPO", ppo_complex_log)]:
    budget_rows.append([name, f"{len(log.rewards)}", f"{log.total_steps:,}",
                        f"{np.mean(log.steps):.0f}"])
display(Markdown("**ComplexEnv training budgets actually used**\n\n"
                 + _md_table(["Algorithm", "Episodes", "Env steps", "Mean steps/ep"],
                             budget_rows)))
