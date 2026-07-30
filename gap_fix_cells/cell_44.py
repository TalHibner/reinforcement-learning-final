class REINFORCEAgent:
    """REINFORCE with Monte-Carlo returns, a selectable baseline, and entropy bonus.

    Objective, per episode of length T (both terms averaged over the same T steps):

        L = -mean_t[ log pi(a_t|s_t) * A_t ]  -  entropy_coef * mean_t[ H(pi(.|s_t)) ]

    The policy term used to be a SUM over steps while the entropy term was a MEAN,
    so the effective entropy weight was ~coef/T, i.e. 1e-5..1e-4 of the policy
    gradient: the advertised exploration bonus was numerically inert. Both terms
    are now means, and `entropy_coef` has its nominal weight.

    Baselines (`baseline=`):
      "running"  (default) A_t = (G_t - mu) / (sigma + 1e-8) with mu, sigma
                 exponential moving statistics of returns *across* episodes. A
                 state-independent constant is a legitimate baseline. Per-episode
                 whitening is not: mu depends on that episode's own actions, and on
                 a zero-information episode (every reward = -step_penalty) it
                 manufactures full-magnitude gradients out of pure discounting
                 geometry, rewarding late actions and punishing early ones for no
                 reason. Running statistics give such an episode a uniformly
                 negative advantage instead, which is the correct signal.
      "episode"  the previous per-episode whitening, kept for the ablation.
      "value"    learned V(s) baseline, A_t = G_t - V(s_t), V fit by MSE to G_t.
                 Unbiased for any state-only baseline, since
                 E[grad log pi(a|s) b(s)] = b(s) grad sum_a pi(a|s) = 0.
      "none"     raw returns.

    Memory: observations and actions are stored, and log-probs/entropies are
    recomputed in a single batched forward pass in update(). The previous version
    kept one live autograd graph per environment step (400 concurrent graphs on
    ComplexEnv); memory is now flat in episode length.

    Truncation: plain MC returns cannot bootstrap, so a time-limit cut leaves a
    finite-horizon bias (bounded by gamma^(T-t) * ||V||_inf per step). With
    `baseline="value"` the tail is bootstrapped with V(s_T) instead; without it the
    bias is accepted and stated.
    """

    def __init__(self, in_channels, n_actions, *,
                 lr=3e-4, gamma=0.99,
                 entropy_coef=0.01,
                 baseline="running",
                 value_lr=None,
                 baseline_momentum=0.02,
                 max_grad_norm=1.0,
                 device="cpu"):
        assert baseline in ("running", "episode", "value", "none")
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.baseline = baseline
        self.baseline_momentum = baseline_momentum
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.n_actions = n_actions

        self.policy = PolicyNetwork(in_channels, n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.value_net = None
        self.value_optimizer = None
        if baseline == "value":
            self.value_net = ValueNetwork(in_channels).to(device)
            self.value_optimizer = torch.optim.Adam(
                self.value_net.parameters(), lr=value_lr or lr)

        self._states = []
        self._actions = []
        self.rewards = []
        self._ret_mean = 0.0
        self._ret_sq = 1.0
        self._ret_seen = 0
        self.last_diagnostics = {}

    # -- helpers -------------------------------------------------------------
    def _tensor(self, obs):
        return torch.as_tensor(np.asarray(obs), dtype=torch.float32,
                               device=self.device).unsqueeze(0)

    def _update_return_stats(self, returns):
        m = float(np.mean(returns))
        sq = float(np.mean(np.square(returns)))
        if self._ret_seen == 0:
            self._ret_mean, self._ret_sq = m, sq
        else:
            a = self.baseline_momentum
            self._ret_mean += a * (m - self._ret_mean)
            self._ret_sq += a * (sq - self._ret_sq)
        self._ret_seen += 1

    def _ret_std(self):
        return float(max(self._ret_sq - self._ret_mean ** 2, 1e-8) ** 0.5)

    # -- interaction ---------------------------------------------------------
    def select_action(self, state, training=True):
        with torch.no_grad():
            logits = self.policy(self._tensor(state))
            if not training:
                return int(logits.argmax(1).item())
            dist = torch.distributions.Categorical(logits=logits)
            action = int(dist.sample().item())

        self._states.append(np.asarray(state, dtype=np.float32))
        self._actions.append(action)
        return action

    # -- learning ------------------------------------------------------------
    def update(self, bootstrap_obs=None):
        """One policy-gradient step on the finished episode.

        bootstrap_obs: the final observation when the episode was cut by the time
        limit. Used only when a learned value baseline is available.
        """
        if len(self.rewards) == 0:
            return None

        G = 0.0
        if bootstrap_obs is not None and self.value_net is not None:
            with torch.no_grad():
                G = float(self.value_net(self._tensor(bootstrap_obs)).item())

        returns = []
        for r in reversed(self.rewards):
            G = float(r) + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        states = torch.as_tensor(np.stack(self._states), dtype=torch.float32,
                                 device=self.device)
        actions = torch.as_tensor(self._actions, dtype=torch.long, device=self.device)

        logits = self.policy(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()

        value_loss_val = float("nan")
        if self.baseline == "value":
            values = self.value_net(states)
            advantages = returns_t - values.detach()
            value_loss = F.mse_loss(values, returns_t)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            self.value_optimizer.step()
            value_loss_val = float(value_loss.item())
        elif self.baseline == "running":
            self._update_return_stats(returns)
            advantages = (returns_t - self._ret_mean) / (self._ret_std() + 1e-8)
        elif self.baseline == "episode":
            advantages = returns_t
            if len(returns_t) > 1:
                advantages = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        else:
            advantages = returns_t

        policy_loss = -(log_probs * advantages).mean()
        entropy_loss = -entropies.mean()
        loss = policy_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.last_diagnostics = {
            "policy_loss": float(policy_loss.item()),
            "entropy": float(entropies.mean().item()),
            "value_loss": value_loss_val,
            "mean_return": float(np.mean(returns)),
        }

        self._states.clear()
        self._actions.clear()
        self.rewards.clear()

        return float(loss.item())


print("REINFORCE agent loaded.")
