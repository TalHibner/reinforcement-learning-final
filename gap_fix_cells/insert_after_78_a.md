### Results tables — generated from this run's evaluation results

The tables below are rendered from `simple_eval_results` / `complex_eval_results`,
so they cannot drift out of step with the run that produced them. Two reporting
choices worth stating:

- **Stage progress is reported as mean and median, not just the maximum.**
  "MaxStage" is a single best episode out of 50; quoting it next to means
  overstates typical behaviour. The full distribution is in the breakdown above.
- **A discounted-return column is included.** Because the true environment reward
  fires only on the terminal goal step, the discounted sparse return of an episode
  is exactly $G_0 = \text{success}\cdot\gamma^{\,\text{steps}-1}$, which is
  computable from the logged episode length and outcome.
