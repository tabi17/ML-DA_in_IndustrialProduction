import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input

# Environment
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n



# DQN Model
class DQN(Model):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc1 = Dense(24, activation='relu')
        self.fc2 = Dense(24, activation='relu')
        self.out = Dense(action_size, activation='linear')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)


model = DQN(action_size)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.MeanSquaredError()


# Hyperparameters
episodes = 500
max_steps = 500
gamma = 0.95  # discount factor
epsilon = 1.0  # exploration probability
epsilon_min = 0.1
epsilon_decay = 0.995

# Training DQN
for ep in range(episodes):
    state = env.reset()[0]
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for step in range(max_steps):
        # Epsilon-greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = model(state).numpy()
            action = np.argmax(q_values[0])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, [1, state_size])
        total_reward += reward

        # Compute target
        target = model(state).numpy()
        if done:
            target[0, action] = reward
        else:
            t = model(next_state).numpy()
            target[0, action] = reward + gamma * np.max(t)

        # Train step
        with tf.GradientTape() as tape:
            q_pred = model(state)
            loss = loss_fn(target, q_pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        state = next_state
        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    if (ep + 1) % 50 == 0:
        print(f"Episode {ep + 1}/{episodes}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")


# Test Trained DQN
state = env.reset()[0]
state = np.reshape(state, [1, state_size])
done = False
total_reward = 0
path = []

while not done:
    action = np.argmax(model(state).numpy()[0])
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    path.append((state.flatten(), action, reward))
    total_reward += reward
    state = np.reshape(next_state, [1, state_size])

print("\nTest Reward:", total_reward)
print("Test Path:", path)
