class DQNAgent:
    """Deep Q-Network agent with target network and epsilon-greedy exploration."""

    def __init__(self, in_channels, n_actions, *,
                 lr=1e-4, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=50_000,
                 buffer_size=10_000, batch_size=32, target_update=1000,
                 train_freq=1, device="cpu"):
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.batch_size = batch_size
        self.target_update = target_update
        self.train_freq = train_freq
        self.device = device

        self.q_net = DQNNetwork(in_channels, n_actions).to(device)
        self.target_net = DQNNetwork(in_channels, n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.steps = 0        # total env steps (drives epsilon and target sync)
        self.last_sync = 0    # env-step count at the last target-network sync
        self.n_syncs = 0

    def get_epsilon(self):
        frac = min(self.steps / self.eps_decay_steps, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state, training=True):
        if training and random.random() < self.get_epsilon():
            return random.randrange(self.n_actions)
        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            return int(self.q_net(t).argmax(1).item())

    def maybe_sync_target(self):
        """Copy online weights into the target net every `target_update` ENV steps.

        The check used to live inside update(), which only runs every `train_freq`
        steps, so the effective period was lcm(target_update, train_freq): the
        sanity config (target_update=10, train_freq=4) synced every 20 steps —
        half the configured rate — and the shipped configs only worked because
        500 and 1000 happen to be divisible by 4. Counting against `self.steps`
        directly decouples the two knobs. Called once per env step by the
        training loop.
        """
        if self.steps - self.last_sync >= self.target_update:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.last_sync = self.steps
            self.n_syncs += 1
            return True
        return False

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s  = torch.as_tensor(s,  dtype=torch.float32, device=self.device)
        a  = torch.as_tensor(a,  dtype=torch.long,    device=self.device)
        r  = torch.as_tensor(r,  dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d  = torch.as_tensor(d,  dtype=torch.float32, device=self.device)

        q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target_net(s2).max(1)[0]
            # d carries terminations only, so a time-limit cut still bootstraps.
            target = r + self.gamma * q_next * (1.0 - d)

        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()


print("DQN agent loaded.")
