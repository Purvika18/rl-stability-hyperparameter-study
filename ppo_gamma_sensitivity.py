import os
import numpy as np
import matplotlib.pyplot as plt

RUNS_DIR = "runs"

def moving_average(x, window=10):
    if window <= 1:
        return x
    return np.convolve(x, np.ones(window)/window, mode="valid")

def load_npy(name):
    path = os.path.join(RUNS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)

def main():
    files = [
        ("gamma=0.90",  "ppo_gamma0.90_seed0.npy"),
        ("gamma=0.99",  "ppo_gamma0.99_seed0.npy"),
        ("gamma=0.999", "ppo_gamma0.999_seed0.npy"),
    ]

    plt.figure(figsize=(10, 6))

    for label, fname in files:
        rewards = load_npy(fname)
        rewards_ma = moving_average(rewards, 10)
        plt.plot(rewards_ma, label=f"{label} (MA10)")

    plt.title("PPO Sensitivity: Gamma (seed=0, lr=3e-4)")
    plt.xlabel("Episode")
    plt.ylabel("Reward (steps survived)")
    plt.grid(True)
    plt.legend()

    out_path = os.path.join(RUNS_DIR, "ppo_gamma_sensitivity.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()

    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
    