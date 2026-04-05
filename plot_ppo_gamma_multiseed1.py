import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from ppo_train import train_ppo

RUNS_DIR = "runs"


def maybe_train(out_path: str, timesteps: int, seed: int, lr: float, gamma: float, clip_range: float, logdir: str, no_clean: bool, verbose: int):
    if os.path.exists(out_path):
        return
    # Use a unique logdir per run so logs don't overwrite
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


def mean_std(curves):
    min_len = min(len(c) for c in curves)
    stacked = np.stack([c[:min_len] for c in curves])
    return stacked.mean(axis=0), stacked.std(axis=0)


def moving_average(x, window=10):
    if window <= 1:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def main():
    p = argparse.ArgumentParser(description="Train (if needed) and plot PPO gamma sensitivity across multiple seeds.")
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds for multi-seed averaging")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--gammas", type=float, nargs="+", default=[0.90, 0.99, 0.999])
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--no-train", action="store_true", help="Only plot; fail if .npy files are missing")
    p.add_argument("--no-clean", action="store_true")
    p.add_argument("--verbose", type=int, default=0)
    args = p.parse_args()

    os.makedirs(RUNS_DIR, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for g in args.gammas:
        curves = []
        for s in args.seeds:
            out_npy = os.path.join(RUNS_DIR, f"ppo_gamma{g}_seed{s}.npy")
            if not args.no_train:
                logdir = os.path.join("ppo_logs", f"gamma{g}_seed{s}")
                maybe_train(
                    out_path=out_npy,
                    timesteps=args.timesteps,
                    seed=s,
                    lr=args.lr,
                    gamma=g,
                    clip_range=args.clip_range,
                    logdir=logdir,
                    no_clean=args.no_clean,
                    verbose=args.verbose,
                )
            if not os.path.exists(out_npy):
                raise FileNotFoundError(f"Missing file: {out_npy}. Run without --no-train to generate it.")
            curves.append(np.load(out_npy))

        m, sd = mean_std(curves)
        m_ma = moving_average(m, args.window)
        sd_ma = moving_average(sd, args.window)
        x = np.arange(len(m_ma))

        plt.plot(x, m_ma, label=f"γ={g} (mean, MA{args.window})")
        plt.fill_between(x, m_ma - sd_ma, m_ma + sd_ma, alpha=0.18)

    plt.title(f"PPO Gamma Sensitivity (mean ± std over seeds {args.seeds})")
    plt.xlabel("Episode")
    plt.ylabel("Reward (steps survived)")
    plt.grid(True)
    plt.legend()

    out_png = os.path.join(RUNS_DIR, "ppo_gamma_multiseed.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.show()

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
