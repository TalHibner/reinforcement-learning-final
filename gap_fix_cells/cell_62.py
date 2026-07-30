# ============================================================
# Policy confidence diagnostic
# ============================================================

def policy_confidence(agent, env_factory, n_episodes=10, seed=42):
    """Print mean max action probability and entropy for a policy agent.

    Built from logits via Categorical(logits=...) — the same numerically stable
    path the agents use, rather than passing softmax probabilities back in.
    """
    max_probs = []
    entropies = []

    env = env_factory()
    for ep in range(n_episodes):
        obs, info = env.reset(seed=episode_seed(seed, ep, "eval"))

        while True:
            t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                if getattr(agent, "policy", None) is not None:
                    logits = agent.policy(t)
                elif getattr(agent, "network", None) is not None:
                    logits, _ = agent.network(t)
                else:
                    raise ValueError("policy_confidence is only for policy agents")

                dist = torch.distributions.Categorical(logits=logits)
                max_probs.append(float(logits.softmax(-1).max().item()))
                entropies.append(float(dist.entropy().item()))
                action = int(logits.argmax(1).item())

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    env.close()

    print(f"Mean max prob: {np.mean(max_probs):.3f}")
    print(f"Mean entropy : {np.mean(entropies):.3f}  "
          f"(uniform over {agent.n_actions} actions = {np.log(agent.n_actions):.3f})")
