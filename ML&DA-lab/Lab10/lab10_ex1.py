import gym
import numpy as np


# Environment
env = gym.make('FrozenLake-v1', is_slippery=False)  # deterministic
state_size = env.observation_space.n
action_size = env.action_space.n


# Q-table initialization

Q = np.zeros((state_size, action_size))


# Hyperparameters
learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.1
episodes = 5000
max_steps = 100


# Training Q-learning
for episode in range(episodes):
    state = env.reset()[0]
    done = False

    for _ in range(max_steps):
        # Epsilon-greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-learning update
        Q[state, action] = Q[state, action] + learning_rate * (
            reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action]
        )

        state = next_state
        if done:
            break


# Testing learned policy
state = env.reset()[0]
done = False
total_reward = 0
path = []

while not done:
    action = np.argmax(Q[state, :])
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward
    path.append((state, action, reward))
    state = next_state

print("Test path:", path)
print("Total reward:", total_reward)

# -----------------------------

print("Optimal actions per episode:")
for s in range(state_size):
    print(f"Episode {s}: Best action = {np.argmax(Q[s, :])}")
