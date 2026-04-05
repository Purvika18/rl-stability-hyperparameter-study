import os
import numpy as np
import matplotlib.pyplot as plt

RUNS_DIR = "runs"

def load_seeds(prefix, seeds):
    curves = []
    for s in seeds:
        path = os.path.join(RUNS_DIR, f"{prefix}{s}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        curves.append(np.load(path))
    return curves

def mean_std(curves):
    min_len = min(len(c) for c in curves)
    stacked = np.stack([c[:min_len] for c in curves])
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return mean, std

def moving_average(x, window=20):
    return np.convolve(x, np.ones(window)/window, mode="valid")

def main():
    seeds = [0, 1, 2, 3, 4]
    curves = load_seeds("ppo_seed", seeds)

    mean, std = mean_std(curves)

    mean_ma = moving_average(mean)
    std_ma = moving_average(std)

    x = np.arange(len(mean_ma))

    plt.figure(figsize=(10, 6))
    plt.plot(x, mean_ma, label="PPO mean (5 seeds)")
    plt.fill_between(x, mean_ma - std_ma, mean_ma + std_ma, alpha=0.25)

    plt.title("PPO Stability Across Random Seeds (mean ± std)")
    plt.xlabel("Episode")
    plt.ylabel("Reward (steps survived)")
    plt.grid(True)
    plt.legend()

    out_path = os.path.join(RUNS_DIR, "ppo_multiseed_mean_std.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()

    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()