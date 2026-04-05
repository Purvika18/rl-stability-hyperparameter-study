import argparse
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# 1) A tiny neural network that outputs action probabilities (left vs right)
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        logits = self.net(x)                 # raw scores
        probs = torch.softmax(logits, dim=-1)  # scores -> probabilities
        return probs


def compute_returns(rewards, gamma: float):
    """Discounted return G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ..."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def set_global_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def reinforce_train(
    env_name="CartPole-v1",
    episodes=500,
    gamma=0.99,
    lr=1e-3,
    print_every=20,
    seed=42,
    hidden_dim=128
):
    env = gym.make(env_name)

    # Seed everything (env + numpy + torch)
    env.reset(seed=seed)
    set_global_seeds(seed)

    obs_dim = env.observation_space.shape[0]   # 4 values for CartPole
    action_dim = env.action_space.n            # 2 actions: left/right

    policy = PolicyNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim, action_dim=action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    all_episode_rewards = []

    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=seed + ep)  # small trick: different start randomness per episode
        done = False

        log_probs = []
        rewards = []

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32)

            # policy -> action distribution
            probs = policy(obs_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))

            obs, reward, terminated, truncated, info = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated

        ep_reward = float(sum(rewards))
        all_episode_rewards.append(ep_reward)

        # compute discounted returns
        returns = compute_returns(rewards, gamma)

        # normalize returns (helps stability)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # REINFORCE loss
        loss = 0.0
        for log_p, Gt in zip(log_probs, returns):
            loss += -log_p * Gt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if print_every > 0 and ep % print_every == 0:
            mean_last = np.mean(all_episode_rewards[-print_every:])
            print(f"[REINFORCE] seed={seed} Episode {ep:4d} | Avg reward (last {print_every}): {mean_last:.2f}")

    env.close()
    return np.array(all_episode_rewards, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=r"runs\reinforce_rewards.npy")
    args = parser.parse_args()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rewards = reinforce_train(
        env_name=args.env,
        episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        print_every=args.print_every,
        seed=args.seed
    )

    np.save(args.out, rewards)

    print("\nFinal check:")
    print(f"[REINFORCE] seed={args.seed} episodes={args.episodes}")
    print(f"Last 10 episode mean reward: {rewards[-10:].mean():.2f}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()