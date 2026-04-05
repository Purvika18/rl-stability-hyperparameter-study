import os
import argparse
import numpy as np
import gymnasium as gym


def run_random(seed: int, episodes: int, out_path: str):
    env = gym.make("CartPole-v1")
    # Seed everything we can
    np.random.seed(seed)
    env.action_space.seed(seed)

    episode_rewards = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)  # different reset seed per episode
        done = False
        total_reward = 0.0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        episode_rewards.append(total_reward)

    env.close()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, np.array(episode_rewards, dtype=np.float32))
    print(f"[Random] seed={seed} episodes={episodes} saved -> {out_path}")
    print(f"[Random] last10 mean={np.mean(episode_rewards[-10:]):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--out", type=str, default=r"runs\random_rewards_seed0.npy")
    args = parser.parse_args()

    run_random(args.seed, args.episodes, args.out)