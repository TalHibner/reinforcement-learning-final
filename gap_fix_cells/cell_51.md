# Training -- SimpleRoomEnv

Sanity check on the empty 10x10 room. Per the lecturer's clarification this
environment does **not** have to be solved to convergence — what has to be visible
is a learning process (return, discounted return, episode length and success rate
vs episode). It is still treated as the diagnostic it was designed to be: DQN and
PPO are expected to clear it quickly, so a greedy success rate well below ~0.9 for
either of them is read as a bug signal in preprocessing, network, hyperparameters
or training loop rather than as a result. REINFORCE is exempt from convergence; its
learning curve is reported as is.

- **Observation:** grayscale (1, 84, 84)
- **Actions:** left / right / forward (Discrete(3))
- **Shaping:** goal x10, step penalty 0.01
- **Model selection:** best-so-far checkpoint by rolling SR100, identical rule for
  all three algorithms (`BestCheckpointer`)
- **Greedy probes:** 5 argmax episodes every 250 episodes, plotted on the
  success-rate panel next to the behaviour-policy curve
