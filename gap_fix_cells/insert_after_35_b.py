# =========================================================
# Reward-shaping mass audit (M8): discounted milestone total vs discounted goal
# =========================================================

# Milestone timings of a *competent* ComplexEnv trajectory, in env steps. These are
# read off the layout: the key is ~30 steps away, the door ~30 more, the water
# cluster sits in the right room, each lava tile costs a fetch-and-toggle round
# trip, and the goal corner is behind the lava ring.
COMPETENT_TIMELINE = {
    "key_bonus":        [30],
    "door_bonus":       [60],
    "right_room_bonus": [65],
    "water_bonus":      [90],
    "lava_bonus":       [110, 140, 170],   # one per lava tile (n_lava = 3)
}
GOAL_STEP = 200        # step at which the goal reward is collected
GAMMA = 0.99


def audit_shaping_mass(shaping=COMPLEX_SHAPING, timeline=COMPETENT_TIMELINE,
                       goal_step=GOAL_STEP, gamma=GAMMA, label="audited config",
                       strict=True):
    """Print and check the shaping budget of a ComplexEnv shaping configuration.

    Returns (discounted_milestone_mass, discounted_goal_payoff).
    """
    rows, disc_total, raw_total = [], 0.0, 0.0
    for name, times in timeline.items():
        bonus = float(shaping[name])
        for t in times:
            disc = bonus * gamma ** t
            rows.append((name, t, bonus, disc))
            disc_total += disc
            raw_total += bonus

    goal_raw = float(shaping["goal_scale"])
    goal_disc = goal_raw * gamma ** goal_step
    step_cost = float(shaping["step_penalty"]) * 400          # full step budget

    print(f"Shaping budget audit — {label} (gamma={gamma})")
    print("-" * 72)
    print(f"{'milestone':<20}{'step':>6}{'bonus':>10}{'discounted':>14}")
    for name, t, bonus, disc in rows:
        print(f"{name:<20}{t:>6}{bonus:>10.2f}{disc:>14.3f}")
    print("-" * 72)
    print(f"{'milestone total':<20}{'':>6}{raw_total:>10.2f}{disc_total:>14.3f}")
    print(f"{'goal payoff':<20}{goal_step:>6}{goal_raw:>10.2f}{goal_disc:>14.3f}")
    print(f"{'ratio milestones/goal':<27}"
          f"{raw_total / goal_raw:>9.2f}x{disc_total / goal_disc:>13.2f}x")
    print(f"\nMilestones-then-idle return (400 steps): "
          f"{raw_total - step_cost:+.2f}   "
          f"(idling is only attractive while this is positive)")
    ok = disc_total < goal_disc
    print("PASS: discounted milestone mass < discounted goal payoff." if ok else
          "FAIL: shaping mass exceeds the goal payoff — the shaped optimum may "
          "exclude reaching the goal.")
    if strict:
        assert ok, ("Shaping budget violated: discounted milestone mass "
                    f"{disc_total:.3f} >= discounted goal payoff {goal_disc:.3f}")
    return disc_total, goal_disc


# The configuration actually used for training.
audit_shaping_mass()

# For the record: the pre-fix configuration this replaces (assert relaxed so the
# cell still runs — it is shown precisely because it fails the test).
print()
audit_shaping_mass(
    shaping=dict(COMPLEX_SHAPING, goal_scale=10.0, key_bonus=1.0, door_bonus=2.0,
                 right_room_bonus=1.0, water_bonus=0.5, lava_bonus=3.0),
    label="pre-fix config (for comparison)",
    strict=False,
)
