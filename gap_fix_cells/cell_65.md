# Training -- ComplexEnv

The hard task: key -> door -> water -> lava -> goal.  Partial progress is expected
and analysed via the 5-stage breakdown.

- **Observation:** colour (3, 84, 84)
- **Actions:** left / right / forward / pickup / drop / toggle (Discrete(6))
- **Shaping:** milestone bonuses (audited against the discounted goal payoff above),
  step penalty 0.01, lava-death penalty 2.0
- **Budget:** 4000 episodes for all three algorithms. REINFORCE additionally carries
  the documented early-stop rule (`EarlyStopper`, patience 1000 episodes) so an
  unequal budget, if it happens, is a measured stop with a reported episode rather
  than an arbitrary cap.
- **Model selection:** best-so-far checkpoint by rolling SR100 for all three
  algorithms, plus greedy ($\epsilon = 0$) probes every 250 episodes so training
  and evaluation measure the same policy.
