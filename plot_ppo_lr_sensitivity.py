import os
import numpy as np
import matplotlib.pyplot as plt

RUNS_DIR = "runs"

def moving_average(x, window=20):
    if window <= 1:
        return x
    return np.convolve(x, np.ones(window)/window, mode="valid")

def load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)

def main():
    # these match what you already generated (seed0 sweep)
    configs = [
        ("lr=3e-5", os.path.join(RUNS_DIR, "ppo_lr3e-5_seed0.npy")),
        ("lr=3e-4", os.path.join(RUNS_DIR, "ppo_lr3e-4_seed0.npy")),
        ("lr=3e-3", os.path.join(RUNS_DIR, "ppo_lr3e-3_seed0.npy")),
    ]

    plt.figure(figsize=(10, 6))

    for label, path in configs:
        y = load(path)
        y_ma = moving_average(y, window=10)
        x = np.arange(len(y_ma))
        plt.plot(x, y_ma, label=label)

    plt.title("PPO Sensitivity: Learning Rate (seed=0)")
    plt.xlabel("Episode (after moving avg)")
    plt.ylabel("Reward (steps survived)")
    plt.grid(True)
    plt.legend()

    out_path = os.path.join(RUNS_DIR, "ppo_lr_sensitivity.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()