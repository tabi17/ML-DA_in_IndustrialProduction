import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False)
Q = np.zeros((env.observation_space.n, env.action_space.n))

learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.1
episodes = 10000
max_steps = 100

for episode in range(episodes):
    state, info, _ = env.reset()          # Gymnasium returns (state, info)
    done = False

    for _ in range(max_steps):

        # Epsilon-greedy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[int(state), :])

        # Step (Gymnasium format)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-learning update
        Q[state, action] = Q[state, action] + learning_rate * (
            reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action]
        )

        state = next_state

        if done:
            break


# TESTARE POLITICĂ ÎNVĂȚATĂ
state, info, _ = env.reset()
done = False
total_reward = 0

while not done:
    action = np.argmax(Q[state, :])
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward
    state = next_state
    env.render()

print("Total reward:", total_reward)
