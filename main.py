import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

def main():
    # training env
    train_env = Monitor(gym.make("CartPole-v1"))

    model = PPO("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=50_000)

    # evaluation env (separate env is best practice)
    eval_env = Monitor(gym.make("CartPole-v1"))
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

    print(f"Mean reward over 10 episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

    model.save("ppo_cartpole")

    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()