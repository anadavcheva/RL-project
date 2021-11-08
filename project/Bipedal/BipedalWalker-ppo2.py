import gym

import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common.policies import MlpPolicy

from stable_baselines import PPO2

if __name__ == '__main__':
    env = gym.make('Humanoid-v2')
    # env = HumanoidEnv()
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
    env = VecNormalize(env)
    # Define the model

    model = PPO2(MlpPolicy, env, n_steps=100, lam=0.95, gamma=0.99, noptepochs=100,
                 ent_coef=0.0, learning_rate=3e-4, cliprange=0.2)

    # Train the agent
    model.learn(total_timesteps=4000)

    obs = np.zeros((env.num_envs,) + env.observation_space.shape)
    obs[:] = env.reset()
    num_episodes = 4000
    for i in range(num_episodes):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Episode: ", i)
        print("Reward:", rewards)
        print("Obs: ", obs)
        print("Info: ", info)
        env.render()
