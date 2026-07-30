# Video Clips

Short clips of each agent at **two points in training**:

1. **Mid-training** — recorded inline during the actual training run, at the episode
   printed in each clip's label: episode 750/2000 (DQN), 1500/3000 (REINFORCE) and
   1250/2500 (PPO) on SimpleRoomEnv, and episode 2000/4000 for all three on
   ComplexEnv. These are snapshots of the *same* agent that continued training to
   produce the final policy, so the pair captures the training trajectory honestly.
2. **After convergence** — greedy evaluation of the selected (best-checkpoint)
   agent.

All six mid-training clips are recorded by the training cells above and displayed
below; the filenames in the display cell are the filenames the training cells write.

### Detailed Failure-Mode Tracking
While the maximum stage reached provides a good high-level view of an agent's progress through ComplexEnv, it doesn't fully capture specific failure modes (e.g. is the agent dying to lava, or just running out of time?). To provide deeper insight into agent behaviour during inference, `evaluate_agent_mode` and the final evaluation table explicitly track **Lava-Death Rate (%)** and the **Average Extinguished Tiles**. This exposes whether an agent fails because it avoids lava entirely or because it attempts to cross it and dies, giving a much clearer picture of its learned policy than the stage breakdown alone.
