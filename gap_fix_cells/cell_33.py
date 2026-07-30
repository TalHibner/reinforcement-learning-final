class SparseReturnTracker(gym.Wrapper):
    """Innermost shim: expose true sparse env reward via info['sparse_reward']."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["sparse_reward"] = float(reward)
        return obs, reward, terminated, truncated, info


class PreprocessWrapper(gym.ObservationWrapper):
    """Resize to (size x size), optional grayscale, HWC -> CHW, normalise to [0,1]."""

    def __init__(self, env, size=84, grayscale=False):
        super().__init__(env)
        self.size = size
        self.grayscale = grayscale
        c = 1 if grayscale else 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(c, size, size), dtype=np.float32)

    def observation(self, obs):
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if obs.ndim == 2:
            obs = obs[:, :, None]
        obs = np.transpose(obs, (2, 0, 1))
        return obs.astype(np.float32) / 255.0


class SimpleShapingWrapper(gym.Wrapper):
    """Scale the sparse goal reward and add a per-step penalty."""

    def __init__(self, env, goal_scale=10.0, step_penalty=0.01):
        super().__init__(env)
        self.goal_scale = goal_scale
        self.step_penalty = step_penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = self.goal_scale * float(reward) - self.step_penalty
        return obs, shaped, terminated, truncated, info


class RichShapingWrapper(gym.Wrapper):
    """Milestone-event reward shaping for ComplexEnv.

    Uses the env getters (curr vs prev) to detect events.
    One-time latched: key pickup, door open, enter right room.
    Per occurrence: water pickup (bounded by ``water_mode``), lava extinguish.
    Also: goal reward scaling, step penalty, lava-death penalty.
    Tracks max_stage reached for analysis.

    Budget discipline (see the shaping-mass audit cell below): the *discounted*
    milestone mass along a competent trajectory must stay strictly below the
    discounted goal payoff, otherwise the shaped optimum is "collect milestones,
    then idle to the step cap" rather than "reach the goal" — which is exactly
    what the previous defaults (13.5 milestone mass vs a goal of 10) produced.
    The defaults below satisfy that inequality by a factor of ~1.8.

    Only *event counts* are used for shaping. No position getter is consulted:
    remaining lava is derived from ``info["n_lava"] - extinguished_lava_count()``,
    because the environment reserves the position getters for analysis.
    """

    def __init__(self, env, goal_scale=20.0, key_bonus=0.5,
                 door_bonus=1.0, right_room_bonus=0.25,
                 water_bonus=0.2, lava_bonus=0.5,
                 step_penalty=0.01, death_penalty=2.0,
                 water_mode="first",
                 drop_water_penalty=0.2):
        super().__init__(env)
        self.goal_scale = goal_scale
        self.key_bonus = key_bonus
        self.door_bonus = door_bonus
        self.right_room_bonus = right_room_bonus
        self.water_bonus = water_bonus
        self.lava_bonus = lava_bonus
        self.step_penalty = step_penalty
        self.death_penalty = death_penalty
        self._key_picked = False
        self._door_opened = False
        self._entered_right = False
        self._ever_picked_water = False
        self._max_stage = 0
        assert water_mode in ("first", "gated", "per_pickup", "per_lava")
        self.water_mode = water_mode
        self._first_water_done = False
        self.drop_water_penalty = drop_water_penalty
        self._water_rewards_given = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._key_picked = False
        self._door_opened = False
        self._entered_right = False
        self._ever_picked_water = False
        self._max_stage = 0
        self._first_water_done = False
        self._water_rewards_given = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        core = self.env.unwrapped
        shaped = self.goal_scale * float(reward) - self.step_penalty

        n_lava = int(info.get("n_lava", 3))
        lava_left = (n_lava - core.extinguished_lava_count()) > 0

        # Key pickup (one-time)
        if not self._key_picked and core.is_carrying_key() and not core.prev_carrying_key():
            shaped += self.key_bonus
            self._key_picked = True

        # Door opened (one-time)
        if not self._door_opened and core.is_door_open() and not core.prev_door_open():
            shaped += self.door_bonus
            self._door_opened = True

        # Entered right room (one-time)
        if not self._entered_right and core.is_in_right_room() and not core.prev_in_right_room():
            shaped += self.right_room_bonus
            self._entered_right = True

        # Water pickup
        picked_water_now = core.is_carrying_water() and not core.prev_carrying_water()

        if picked_water_now:
            self._ever_picked_water = True
            give = False

            current_extinguished = core.extinguished_lava_count()
            max_water_rewards = n_lava

            if self.water_mode == "first":
                # Reward only the first water pickup in the episode.
                if not self._first_water_done:
                    give = True
                    self._first_water_done = True

            elif self.water_mode == "gated":
                # Old behaviour: can still be farmed by repeated pickup/drop.
                if self._door_opened and lava_left:
                    give = True

            elif self.water_mode == "per_pickup":
                # Fully permissive, mostly for ablation/debugging.
                give = True

            elif self.water_mode == "per_lava":
                # Anti-farming mode:
                # allow one water-pickup bonus for each lava-progress level.
                #
                # Before any lava is extinguished:
                #   current_extinguished = 0, water_rewards_given = 0 -> reward first water pickup.
                #
                # Repeated pickup/drop before extinguishing:
                #   current_extinguished = 0, water_rewards_given = 1 -> no reward.
                #
                # After one lava tile is extinguished:
                #   current_extinguished = 1, water_rewards_given = 1 -> reward next water pickup.
                if (
                    self._door_opened
                    and lava_left
                    and self._water_rewards_given <= current_extinguished
                    and self._water_rewards_given < max_water_rewards
                ):
                    give = True
                    self._water_rewards_given += 1

            if give:
                shaped += self.water_bonus

        # Penalise dropping water before extinguishing lava.
        # This discourages water pickup/drop loops and encourages carrying water to lava.
        # The action id is read from the env's own action enum so the check keeps
        # working if the action subset or wrapper order changes.
        dropped_water_now = (
            core.prev_carrying_water()
            and not core.is_carrying_water()
            and int(action) == int(core.actions.drop)
        )

        if dropped_water_now and lava_left:
            shaped -= self.drop_water_penalty

        # Lava extinguished (per tile)
        if core.extinguished_lava_count() > core.prev_extinguished_lava_count():
            shaped += self.lava_bonus

        # Lava death penalty
        if info.get("died_on_lava", False):
            shaped -= self.death_penalty

        # Stage tracking for analysis
        stage = 0
        if self._key_picked:
            stage = 1
        if self._door_opened:
            stage = 2
        if self._ever_picked_water:
            stage = 3
        if core.extinguished_lava_count() > 0:
            stage = 4
        if info.get("is_success", False):
            stage = 5
        self._max_stage = max(self._max_stage, stage)
        info["stage_reached"] = self._max_stage

        return obs, shaped, terminated, truncated, info


print("Wrappers loaded.")
