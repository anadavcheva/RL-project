from collections import deque
import gym
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common.policies import MlpPolicy

from stable_baselines import PPO2
import matplotlib.pyplot as plt
if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3')
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
    env = VecNormalize(env)
    # Define the model

    model = PPO2(MlpPolicy, env, n_steps=100, lam=0.95, gamma=0.99, noptepochs=100,
                 ent_coef=0.0, learning_rate=3e-4, cliprange=0.2)

    # Train the agent
    model.learn(total_timesteps=4000)

    deck = deque(maxlen=2000)
    ep_rewards = []

    rewards_over_time = []
    error = []
    epsilon = []
    rew_var = []
    rew_mean = []
    mean_100 = []

    num_episodes = 100

    for i in range(num_episodes):
        ep_rewards = []
        observation = np.zeros((env.num_envs,) + env.observation_space.shape)
        observation[:] = env.reset()
        while True:
            env.render()
            action, _states = model.predict(observation)
            obs, rewards, flag, info = env.step(action)
            ep_rewards.append(rewards[0])
            observation = obs
            ep_rew_total = sum(ep_rewards)
            mean = np.mean(ep_rewards)
            var = np.var(ep_rewards)
            if ep_rew_total < -300:
                flag = True

            if flag:
                rewards_over_time.append(ep_rew_total)
                rew_mean.append(mean)
                rew_var.append(var)
                max_reward = np.max(rewards_over_time)
                episode_max = np.argmax(rewards_over_time)
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("Episode: ", i)
                print("Reward:", rewards)

                break

    plt.figure(1)
    plt.plot(rewards_over_time, label="Rewards")
    plt.plot(rew_mean, label="Mean")
    plt.plot(rew_var, label="Variance")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Rewards per Episode")
    plt.legend(loc=0)
    plt.show()