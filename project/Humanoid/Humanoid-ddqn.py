
import numpy as np

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from Bipedal.deep_q_learning import DDQN
from Humanoid.HumanoidEnv import HumanoidEnv


def build_model(state_size, action_size, learning_rate):
    model = Sequential()

    model.add(Dense(24, input_dim= state_size, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(24, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(action_size, activation='linear',
                    kernel_initializer='he_uniform'))
    model.summary()
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    return model


if __name__ == '__main__':
    env = HumanoidEnv()

    state_space_shape = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    learning_rate = 0.01
    discount_factor = 0.99
    epsilon = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.01

    batch_size = 64
    memory_size = 1024

    model = build_model(state_space_shape, num_actions, learning_rate)
    target_model = build_model(state_space_shape, num_actions, learning_rate)

    agent = DDQN(state_space_shape, num_actions, model, target_model, learning_rate, discount_factor, batch_size, memory_size)

    # agent.update_memory()

    agent.train()

    scores, episodes = [], []
    num_episodes = 300

    for episode in range(num_episodes):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_space_shape])

        while not done:
            env.render()

            action = agent.get_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_space_shape])
            reward = reward if not done or score == 499 else -100
            if epsilon > epsilon_min:
                epsilon -= (epsilon - epsilon_min) / num_episodes

            agent.append_sample(state, action, reward, next_state, done)
            # if len(agent.memory != 0 and agent.train_start is not None):
            #     if len(agent.memory) >= agent.train_start:
            agent.train_model()
            score += reward
            state = next_state

            if done:
                agent.update_target_model()
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(episode)