import argparse
import os
import shutil

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy


def train_ppo(
    total_timesteps=50_000,
    seed=0,
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.0,
    clip_range=0.2,
    log_dir="ppo_logs",
    out_path="runs/ppo_rewards_seed0.npy",
    clean_logs=True,
    verbose=1,
):
    # 1) Ensure fresh logs (IMPORTANT so we don’t mix older runs)
    if clean_logs and os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # 2) Create env: Monitor + VecEnv + seed
    env = make_vec_env(
        "CartPole-v1",
        n_envs=1,
        seed=seed,
        monitor_dir=log_dir,   # writes Monitor.csv here
    )

    # 3) Create PPO model with tunable hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=ent_coef,
        clip_range=clip_range,
        verbose=verbose,
        seed=seed,
    )

    # 4) Train
    model.learn(total_timesteps=total_timesteps)

    # 5) Read episode rewards from Monitor logs and save them
    x, y = ts2xy(load_results(log_dir), "timesteps")  # y = episode rewards

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(out_path, y.astype(np.float32))

    print(f"\nSaved PPO episode rewards to: {out_path}")
    print(f"Total PPO episodes logged: {len(y)}")

    env.close()
    return y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--logdir", type=str, default="ppo_logs")
    p.add_argument("--out", type=str, default=r"runs\ppo_rewards_seed0.npy")
    p.add_argument("--no-clean", action="store_true", help="Do not delete old logdir before training")
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args()

    train_ppo(
        total_timesteps=args.timesteps,
        seed=args.seed,
        learning_rate=args.lr,
        gamma=args.gamma,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        log_dir=args.logdir,
        out_path=args.out,
        clean_logs=(not args.no_clean),
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()