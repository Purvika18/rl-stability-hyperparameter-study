import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, window=20):
    x = np.array(x, dtype=np.float32)
    return np.convolve(x, np.ones(window)/window, mode="valid")

rewards = np.load("reinforce_rewards.npy")

plt.figure()
plt.plot(rewards, label="REINFORCE reward per episode")
plt.plot(moving_average(rewards, 20), label="REINFORCE moving avg (20)")
plt.xlabel("Episode")
plt.ylabel("Reward (steps survived)")
plt.title("REINFORCE on CartPole-v1")
plt.legend()
plt.show()