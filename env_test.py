import gymnasium as gym
env = gym.make("CartPole-v1")
observation, info = env.reset()
print ("Intial observation:", observation)

for step in range(5):
    action = env.action_space.sample()
    observation, reward, terminated, truncated , info = env.step(action)

    print(f"Step {step +1}")
    print("Action:", action)
    print("Observation:", observation)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    if terminated or truncated:
        print("Episode ended.")
        break

env.close
