## DQN Agent

Value-based method with:
- Q-network + periodic target-network sync, on an **environment-step clock**
  (`maybe_sync_target`) so the sync period is independent of `train_freq`
- Linear epsilon-greedy decay, with a schedule length never longer than the run
  and periodic greedy ($\epsilon = 0$) probes during training
- Huber (smooth-L1) loss, gradient clipping
- Bellman targets that bootstrap through time-limit truncations
  ($d$ carries terminations only)
