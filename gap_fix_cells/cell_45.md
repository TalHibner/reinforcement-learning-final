## PPO Agent

Proximal Policy Optimisation with:
- Shared CNN backbone, separate actor + critic heads, **orthogonal initialisation**
  (trunk gain $\sqrt2$, policy head gain $0.01$ so the initial policy is
  near-uniform with entropy $\approx \ln|\mathcal{A}|$, value head gain $1$)
- GAE($\lambda$) for advantage estimation, with **time limits handled as
  truncations, not terminations**: $\delta_T = r_T + \gamma V(s_{T+1}) - V(s_T)$ at
  a time limit, $\delta_T = r_T - V(s_T)$ only at a true terminal, and the trace cut
  at both (Pardo et al., 2018)
- Clipped surrogate objective ($\epsilon = 0.2$) over **shuffled minibatches**,
  advantages normalised per minibatch
- KL early stopping using the low-variance estimator
  $\widehat{KL} = (r - 1) - \log r$, checked **before** the optimiser step
- Linear learning-rate annealing over training
- Per-update telemetry (approx-KL, clip fraction, entropy, explained variance)
  printed with the periodic progress line

**Value-loss clipping is deliberately omitted.** The evidence in the PPO
implementation literature (Engstrom et al., 2020) is that it does not reliably
help and can hurt; the returns here are already on a bounded scale, so there is
nothing it needs to protect.
