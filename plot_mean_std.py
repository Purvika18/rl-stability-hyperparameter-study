import os
import numpy as np
import matplotlib.pyplot as plt

RUNS_DIR = "runs"  # your folder

def load_seeds(prefix, seeds=(0,1,2)):
    """
    Loads runs/<prefix>_seed{seed}.npy for all seeds.
    Returns a list of 1D arrays (episode rewards).
    """
    curves = []
    for s in seeds:
        path = os.path.join(RUNS_DIR, f"{prefix}_seed{s}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        curves.append(np.load(path))
    return curves

def mean_std_curve(curves):
    """
    Curves have different lengths (different #episodes).
    We cut all to the shortest so we can compute mean/std.
    """
    min_len = min(len(c) for c in curves)
    cut = np.stack([c[:min_len] for c in curves], axis=0)  # (seeds, episodes)
    mean = cut.mean(axis=0)
    std = cut.std(axis=0)
    return mean, std, min_len

def moving_average(x, window=20):
    if window <= 1:
        return x
    out = np.convolve(x, np.ones(window)/window, mode="valid")
    return out

def plot_with_shading(mean, std, label, window=20):
    m = moving_average(mean, window)
    s = moving_average(std, window)
    x = np.arange(len(m))
    plt.plot(x, m, label=label)
    plt.fill_between(x, m - s, m + s, alpha=0.2)

def main():
    # Load curves
    rand_curves = load_seeds("random_rewards")
    reinf_curves = load_seeds("reinforce_rewards")
    ppo_curves = load_seeds("ppo_rewards")

    # Mean/std
    rand_mean, rand_std, _ = mean_std_curve(rand_curves)
    reinf_mean, reinf_std, _ = mean_std_curve(reinf_curves)
    ppo_mean, ppo_std, _ = mean_std_curve(ppo_curves)

    plt.figure(figsize=(10, 6))
    plot_with_shading(rand_mean, rand_std, "Random (mean ± std)", window=20)
    plot_with_shading(reinf_mean, reinf_std, "REINFORCE (mean ± std)", window=20)
    plot_with_shading(ppo_mean, ppo_std, "PPO (mean ± std)", window=20)

    plt.title("CartPole-v1: Random vs REINFORCE vs PPO (3 seeds)")
    plt.xlabel("Episode (after moving avg)")
    plt.ylabel("Reward (steps survived)")
    plt.grid(True)
    plt.legend()

    out_path = os.path.join(RUNS_DIR, "compare_mean_std.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()