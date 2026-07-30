### Shaping-budget audit — does the shaping preserve the goal as the optimum?

Event shaping is only safe if the agent cannot do better by farming milestones than
by finishing the task. The decisive comparison is not the *raw* bonus total but the
**discounted** total at decision time: milestones fire early, the goal fires ~200
steps later, and at $\gamma = 0.99$ a payoff 200 steps away is worth only
$\gamma^{200} \approx 0.13$ of its face value.

The cell below computes both sides of the inequality along an explicit competent
trajectory (the milestone timings a solving agent would hit) and asserts

$$\sum_i \gamma^{t_i} b_i \;<\; \gamma^{T_{goal}} \cdot \text{goal\_scale}$$

The earlier configuration (`goal_scale=10`, `key=1`, `door=2`, `right_room=1`,
`water=0.5`, `lava=3`/tile) fails this test by 3.6x: **+13.5** of milestone mass against
a goal worth **+10** undiscounted and **+1.34** discounted — so "collect the milestones,
then idle until the step cap" was genuinely the better shaped policy, and the trained
PPO agent found exactly that optimum (key → door → water, then 400 steps of nothing).
The audited configuration keeps every milestone as a learning signal but places the
whole ladder strictly below the goal payoff.
