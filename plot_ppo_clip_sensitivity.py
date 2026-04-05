import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from ppo_train import train_ppo

RUNS_DIR = "runs"


def moving_average(x, window=10):
    if window <= 1:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def maybe_train(out_path: str, timesteps: int, seed: int, lr: float, gamma: float, clip_range: float, logdir: str, no_clean: bool, verbose: int):
    if os.path.exists(out_path):
        return
    train_ppo(
        total_timesteps=timesteps,
        seed=seed,
        learning_rate=lr,
        gamma=gamma,
        clip_range=clip_range,
        log_dir=logdir,
        out_path=out_path,
        clean_logs=(not no_clean),
        verbose=verbose,
    )


def main():
    p = argparse.ArgumentParser(description="Train (if needed) and plot PPO clip-range sensitivity.")
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--no-train", action="store_true", help="Only plot; fail if .npy files are missing")
    p.add_argument("--no-clean", action="store_true", help="Do not delete old PPO logdir when training")
    p.add_argument("--verbose", type=int, default=0)
    args = p.parse_args()

    os.makedirs(RUNS_DIR, exist_ok=True)

    clip_values = [0.1, 0.2, 0.4]
    curves = []

    for eps in clip_values:
        out_npy = os.path.join(RUNS_DIR, f"ppo_clip{eps}_seed{args.seed}.npy")
        if not args.no_train:
            maybe_train(
                out_path=out_npy,
                timesteps=args.timesteps,
                seed=args.seed,
                lr=args.lr,
                gamma=args.gamma,
                clip_range=eps,
                logdir="ppo_logs",
                no_clean=args.no_clean,
                verbose=args.verbose,
            )
        if not os.path.exists(out_npy):
            raise FileNotFoundError(f"Missing file: {out_npy}. Run without --no-train to generate it.")
        rewards = np.load(out_npy)
        curves.append((eps, rewards))

    plt.figure(figsize=(10, 6))
    for eps, rewards in curves:
        plt.plot(moving_average(rewards, args.window), label=f"ε={eps} (MA{args.window})")

    plt.title(f"PPO Clip Range Sensitivity (seed={args.seed}, γ={args.gamma}, lr={args.lr})")
    plt.xlabel("Episode")
    plt.ylabel("Reward (steps survived)")
    plt.grid(True)
    plt.legend()

    out_png = os.path.join(RUNS_DIR, "ppo_clip_sensitivity.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.show()

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
