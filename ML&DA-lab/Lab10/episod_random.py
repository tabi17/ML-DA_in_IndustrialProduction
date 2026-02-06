#test
import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False)
state, _ = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    print("State:", state, " Reward:", reward)
