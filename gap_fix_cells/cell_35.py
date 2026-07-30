def make_simple_env(max_steps=100):
    """Wrapped SimpleRoomEnv: grayscale, 3 actions, goal reward x10."""
    env = SimpleRoomEnv(max_steps=max_steps)
    env = SparseReturnTracker(env)
    env = SimpleShapingWrapper(env, goal_scale=10.0, step_penalty=0.01)
    env = ActionSubsetWrapper(env, action_ids=(0, 1, 2))  # left, right, forward
    env = PreprocessWrapper(env, size=84, grayscale=True)
    return env


# ComplexEnv shaping constants, stated once here and audited in the next cell
# against the discounted goal payoff (see M8 / "Reward Shaping Analysis").
COMPLEX_SHAPING = dict(
    goal_scale=20.0,
    key_bonus=0.5,
    door_bonus=1.0,
    right_room_bonus=0.25,
    water_bonus=0.2,
    lava_bonus=0.5,
    step_penalty=0.01,
    death_penalty=2.0,
    water_mode="first",
    drop_water_penalty=0.2,
)


def make_complex_env(max_steps=400):
    """Wrapped ComplexEnv: colour, 6 actions, rich milestone shaping."""
    env = ComplexEnv(max_steps=max_steps)
    env = SparseReturnTracker(env)
    env = RichShapingWrapper(env, **COMPLEX_SHAPING)
    env = ActionSubsetWrapper(env, action_ids=(0, 1, 2, 3, 4, 5))
    env = PreprocessWrapper(env, size=84, grayscale=False)
    return env


# Quick sanity check
tmp_env = make_complex_env()
w = tmp_env

while hasattr(w, "env"):
    if isinstance(w, RichShapingWrapper):
        print(
            "ComplexEnv shaping:",
            "water_mode =", w.water_mode,
            "water_bonus =", w.water_bonus,
            "lava_bonus =", w.lava_bonus,
            "goal_scale =", w.goal_scale,
        )
        break
    w = w.env

tmp_env.close()

# Quick sanity check
_ts = make_simple_env(); _tc = make_complex_env()
_os, _ = _ts.reset(seed=0); _oc, _ = _tc.reset(seed=0)
print(f"SimpleRoomEnv  obs: {_os.shape} {_os.dtype}  actions: {_ts.action_space}")
print(f"ComplexEnv     obs: {_oc.shape} {_oc.dtype}  actions: {_tc.action_space}")
_ts.close(); _tc.close()
del _ts, _tc, _os, _oc
