class PPOAgent:
    """PPO-Clip with GAE, shuffled minibatches, entropy bonus and KL early-stop.

    Three departures from the previous version, all of them standard practice
    (Schulman et al., 2017; Engstrom et al., 2020; Huang et al., 2022):

    1. **Time limits are not terminals.** The TD residual bootstraps through a
       truncation, delta_T = r_T + gamma*V(s_T+1) - V(s_T), and only the GAE trace
       is cut there; a true termination (goal / lava) still uses V = 0. Treating
       truncation as termination told the critic the world ends with value 0 at
       max_steps, so the same pixel state seen at t=10 and t=390 carried different
       targets — an irreducible noise floor in the value regression, and therefore
       in every advantage (Pardo et al., 2018). On ComplexEnv, where essentially
       every episode truncates, that was 100% of episode boundaries.
    2. **Shuffled minibatches.** n_epochs passes over the rollout in
       n_minibatches shuffled chunks with per-minibatch advantage normalisation,
       instead of a handful of full-batch steps.
    3. **KL early-stop before the step.** approx_kl uses Schulman's low-variance
       unbiased estimator k3 = (r-1) - log r and is checked *before*
       optimizer.step(), so no update is applied past the threshold. The previous
       version broke out only after the step had already landed.

    Value-loss clipping is deliberately omitted: the evidence (Engstrom et al.,
    2020) is that it does not help and can hurt, and with GAE returns already
    normalised in scale there is nothing it needs to protect here.
    """

    def __init__(self, in_channels, n_actions, *,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_ratio=0.2, value_coef=0.5,
                 entropy_coef=0.01, max_grad_norm=0.5,
                 n_epochs=4, n_steps=128, n_minibatches=4,
                 target_kl=None, anneal_lr=True,
                 device="cpu"):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.n_minibatches = max(1, int(n_minibatches))
        self.target_kl = target_kl
        self.anneal_lr = anneal_lr
        self.lr0 = lr
        self.device = device
        self.n_actions = n_actions

        self.network = ActorCriticNetwork(in_channels, n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = RolloutBuffer()
        self.last_diagnostics = {}

    def select_action(self, state, training=True):
        t = torch.as_tensor(state, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        with torch.no_grad():
            if not training:
                logits, _ = self.network(t)
                return int(logits.argmax(1).item())
            action, lp, _, val = self.network.get_action_and_value(t)
            return int(action.item()), float(lp.item()), float(val.item())

    def get_value(self, state):
        """V(s) for a single observation — used for truncation bootstrapping."""
        t = torch.as_tensor(state, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, v = self.network(t)
        return float(v.item())

    def set_lr_fraction(self, frac):
        """Linear learning-rate anneal: lr <- lr0 * frac (1 at start, 0 at end)."""
        if not self.anneal_lr:
            return
        lr = self.lr0 * float(np.clip(frac, 0.0, 1.0))
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def compute_gae(self, next_value):
        rewards = self.buffer.rewards
        values = self.buffer.values
        terminateds = self.buffer.terminateds
        truncateds = self.buffer.truncateds
        boundary_values = self.buffer.boundary_values

        T = len(rewards)
        adv = [0.0] * T
        gae = 0.0
        for t in reversed(range(T)):
            if terminateds[t]:
                next_v, trace = 0.0, 0.0          # true terminal: no future value
            elif truncateds[t]:
                next_v, trace = boundary_values[t], 0.0   # time limit: bootstrap, cut trace
            elif t == T - 1:
                next_v, trace = next_value, 1.0   # rollout cut mid-episode
            else:
                next_v, trace = values[t + 1], 1.0
            delta = rewards[t] + self.gamma * next_v - values[t]
            gae = delta + self.gamma * self.gae_lambda * trace * gae
            adv[t] = gae

        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        val_t = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        return adv_t, adv_t + val_t

    def update(self, next_value):
        advantages, returns = self.compute_gae(next_value)

        states = torch.as_tensor(
            np.array(self.buffer.states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(
            self.buffer.actions, dtype=torch.long, device=self.device)
        old_lp = torch.as_tensor(
            self.buffer.log_probs, dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(
            self.buffer.values, dtype=torch.float32, device=self.device)

        batch_size = states.size(0)
        minibatch_size = max(1, batch_size // self.n_minibatches)

        total_loss, n_updates = 0.0, 0
        kls, clip_fracs, entropies = [], [], []
        stop_early = False

        for _ in range(self.n_epochs):
            perm = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                mb = perm[start:start + minibatch_size]

                _, new_lp, entropy, new_val = self.network.get_action_and_value(
                    states[mb], actions[mb])
                new_val = new_val.squeeze(-1)

                logratio = new_lp - old_lp[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = float(((ratio - 1.0) - logratio).mean().item())
                    clip_fracs.append(float(
                        ((ratio - 1.0).abs() > self.clip_ratio).float().mean().item()))
                    entropies.append(float(entropy.mean().item()))

                # Early-stop BEFORE applying this step, not after.
                if self.target_kl is not None and approx_kl > self.target_kl:
                    stop_early = True
                    break
                kls.append(approx_kl)

                mb_adv = advantages[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio,
                                    1 + self.clip_ratio) * mb_adv

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_val, returns[mb])
                entropy_loss = -entropy.mean()

                loss = (policy_loss
                        + self.value_coef * value_loss
                        + self.entropy_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += float(loss.item())
                n_updates += 1

            if stop_early:
                break

        # Critic quality on this rollout, measured with the pre-update values.
        with torch.no_grad():
            var_y = float(returns.var().item())
            ev = float("nan") if var_y == 0 else float(
                1.0 - ((returns - old_values).var() / var_y).item())

        self.last_diagnostics = {
            "gradient_steps": n_updates,
            "approx_kl": float(np.mean(kls)) if kls else 0.0,
            "clip_frac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "explained_variance": ev,
            "early_stopped": stop_early,
        }

        self.buffer.clear()
        return total_loss / max(1, n_updates)


print("PPO agent loaded.")
