import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, window=20):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode="valid")


reinforce = np.load("reinforce_rewards.npy")
ppo = np.load("ppo_rewards.npy")

plt.figure(figsize=(10, 6))

plt.plot(moving_average(reinforce, 20), label="REINFORCE (moving avg)")
plt.plot(moving_average(ppo, 20), label="PPO (moving avg)")

# Random agent baseline as horizontal line
plt.axhline(y=20.5, color="red", linestyle="--", label="Random agent (~20 reward)")

plt.xlabel("Episode")
plt.ylabel("Reward (steps survived)")
plt.title("CartPole-v1: Random vs REINFORCE vs PPO")
plt.legend()
plt.grid(True)
plt.show()