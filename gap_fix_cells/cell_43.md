## REINFORCE Agent

Policy-gradient method with:
- Episode-level Monte Carlo returns
- A **running-statistics baseline** across episodes (`baseline="running"`), with
  per-episode whitening (`"episode"`) and a learned $V(s)$ baseline (`"value"`)
  available for the ablation
- Entropy regularisation for exploration, on the **same scale** as the policy term
  (both averaged over the episode's steps, so `entropy_coef` means what it says)
- Observations stored and log-probabilities recomputed in one batched pass, so
  memory does not grow with episode length

Loss actually optimised, for an episode of length $T$:

$$L = -\frac{1}{T}\sum_{t=1}^{T}\log \pi(a_t \mid s_t)\,A_t \; - \; c_H \cdot \frac{1}{T}\sum_{t=1}^{T} H\big(\pi(\cdot \mid s_t)\big)$$

with $A_t = (G_t - \mu)/(\sigma + \varepsilon)$ under running statistics
$\mu, \sigma$. Truncated episodes keep finite-horizon returns (bias bounded by
$\gamma^{T-t}\lVert V\rVert_\infty$) unless the learned baseline is enabled, in
which case the tail is bootstrapped with $V(s_T)$.
