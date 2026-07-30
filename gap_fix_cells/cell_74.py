# Sanity check before evaluation: confirm which weights are about to be evaluated.
print("DQN q_net first-layer weight sum:",
      dqn_complex_agent.q_net.features.conv[0].weight.abs().sum().item())

# The same best-so-far rule is applied to all three algorithms. (When the notebook
# was run with LOAD_FROM_ARTIFACTS=True the checkpointers do not exist in this
# session — the weights were loaded straight from the saved best checkpoints.)
for label, var in (("DQN", "dqn_complex_ckpt"),
                   ("REINFORCE", "reinforce_complex_ckpt"),
                   ("PPO", "ppo_complex_ckpt")):
    ckpt = globals().get(var)
    if ckpt is None:
        print(f"{label:<10} loaded from artifacts (no checkpointer in this session)")
    else:
        print(f"{label:<10} best episode {ckpt.best_episode}, "
              f"score (SR100, mean stage, mean shaped) = {ckpt.best_score}")
