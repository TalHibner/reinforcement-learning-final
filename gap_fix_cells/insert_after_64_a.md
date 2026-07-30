### Ablation — what baseline should REINFORCE use?

The Discussion previously claimed a value-baseline ablation that did not exist in
the notebook. This is that ablation, run for real, on a matched budget and seed.
Three variants of the same agent, differing only in how the advantage is formed:

| Variant | Advantage $A_t$ | Why it is in the comparison |
|---|---|---|
| `episode` | $(G_t - \mu_{ep}) / (\sigma_{ep} + \varepsilon)$ | The original implementation. Not a valid baseline: $\mu_{ep}$ depends on that episode's own actions, and on an all-failure episode (every reward $= -0.01$) it produces a full-magnitude gradient purely from discounting geometry. |
| `running` | $(G_t - \mu) / (\sigma + \varepsilon)$, EMA across episodes | A state-independent constant *is* a valid baseline; a zero-information episode now gets a uniformly negative advantage instead of an arbitrary within-episode ranking. This is the default used for the reported runs. |
| `value` | $G_t - \hat V(s_t)$, $\hat V$ trained by MSE on $G_t$ | The textbook variance reduction. Unbiased for any state-only baseline, since $\mathbb{E}[\nabla \log \pi(a\mid s)\, b(s)] = b(s) \nabla \sum_a \pi(a \mid s) = 0$, and $b^*(s) \approx V^\pi(s)$ is near variance-optimal. |

Budget is deliberately short (`ABLATION_EPISODES`) — this is a comparison between
variants under identical conditions, not an attempt to solve the task. The numbers
printed by the cell are what the Discussion refers to; nothing is asserted about the
outcome ahead of the run.
